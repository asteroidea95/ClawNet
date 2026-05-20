"""
母模型推理服务 - 轻量 HTTP API

在 GTX 1060 4GB 上稳定运行，峰值显存 ~300MB

用法:
  # 启动服务
  python serve.py --checkpoint ./checkpoints/mother_int4.pt
  
  # 调用 API
  curl -X POST http://localhost:8888/analyze \
    -H "Content-Type: application/json" \
    -d '{
      "intent": "看看这段代码有没有问题",
      "code": "func main() { fmt.Println(undefinedVar) }",
      "language": "go"
    }'
"""

import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MotherModelConfig
from model.mother_model import MotherModel
from tokenizer.code_tokenizer import CodeTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mother_model")


class MotherModelServer:
    """母模型推理服务"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu", use_quantized: bool = True):
        self.device = torch.device(device)
        self.config = MotherModelConfig()
        self.tokenizer = CodeTokenizer()
        
        logger.info(f"Loading model from {checkpoint_path}...")
        logger.info(f"Device: {device}")
        
        self.model = MotherModel(self.config)
        
        if checkpoint_path:
            data = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in data:
                self.model.load_state_dict(data['model_state_dict'])
            elif 'quantized' in data:
                logger.info("Loading quantized weights...")
                for name, qdata in data['quantized'].items():
                    self._load_quantized_weight(name, qdata)
                for name, param in data['non_quantized'].items():
                    self._set_param(name, param)
            else:
                self.model.load_state_dict(data)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        total_params = self.model.get_total_params()
        logger.info(f"Model loaded: {total_params/1e6:.1f}M parameters")
    
    def _load_quantized_weight(self, name: str, qdata: Dict):
        """加载量化后的权重"""
        from inference.quantize import QuantizedLinear
        
        module = self.model
        parts = name.split('.')
        for p in parts[:-1]:
            module = getattr(module, p)
        
        if hasattr(module, parts[-1]):
            old_module = getattr(module, parts[-1])
            if isinstance(old_module, torch.nn.Linear):
                quantized_linear = QuantizedLinear(qdata, old_module)
                setattr(module, parts[-1], quantized_linear)
    
    def _set_param(self, name: str, param: torch.Tensor):
        """设置参数"""
        module = self.model
        parts = name.split('.')
        for p in parts[:-1]:
            module = getattr(module, p)
        if hasattr(module, parts[-1]):
            getattr(module, parts[-1]).data = param.to(self.device)
    
    @torch.no_grad()
    def analyze(
        self,
        intent: str,
        code: str,
        language: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        分析代码并返回诊断结果
        
        Args:
            intent: 用户意图（如"看看这段代码有问题没"）
            code: 代码内容
            language: 语言（go/python/ts）
            max_tokens: 最大生成长度
            temperature: 采样温度
        
        Returns:
            {
                "status": "ok",
                "diagnosis": "诊断描述",
                "culprit_submodel": "go",
                "intent_vector": [...],
                "code_vector": [...],
            }
        """
        # 编码意图
        intent_ids = self._encode_text(intent, max_len=256)
        intent_tensor = torch.tensor([intent_ids], dtype=torch.long, device=self.device)
        intent_mask = torch.tensor([[i != 0 for i in intent_ids]], dtype=torch.bool, device=self.device)
        
        # 编码代码
        code_ids, depth_ids, sibling_ids = self.tokenizer.encode(code)
        max_code_len = 2048
        if len(code_ids) > max_code_len:
            code_ids = code_ids[:max_code_len]
            depth_ids = depth_ids[:max_code_len]
            sibling_ids = sibling_ids[:max_code_len]
        
        pad_len = max_code_len - len(code_ids)
        code_ids += [0] * pad_len
        depth_ids += [0] * pad_len
        sibling_ids += [0] * pad_len
        
        code_tensor = torch.tensor([code_ids], dtype=torch.long, device=self.device)
        depth_tensor = torch.tensor([depth_ids], dtype=torch.long, device=self.device)
        sibling_tensor = torch.tensor([sibling_ids], dtype=torch.long, device=self.device)
        code_mask = torch.tensor([[i != 0 for i in code_ids]], dtype=torch.bool, device=self.device)
        
        # 编码起始 token [bos]
        start_ids = torch.tensor([[1]], dtype=torch.long, device=self.device)  # <bos>
        
        # 自回归生成
        generated = start_ids.clone()
        eos_token_id = 2  # <eos>
        
        for _ in range(max_tokens):
            outputs = self.model(
                intent_input_ids=intent_tensor,
                intent_mask=intent_mask,
                code_input_ids=code_tensor,
                code_depth_ids=depth_tensor,
                code_sibling_ids=sibling_tensor,
                code_mask=code_mask,
                target_ids=generated,
            )
            
            logits = outputs['logits']
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            
            # 采样
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() == eos_token_id:
                break
        
        # 解码
        output_ids = generated[0].tolist()
        diagnosis = self.tokenizer.decode(output_ids)
        
        return {
            "status": "ok",
            "diagnosis": diagnosis,
            "language": language,
        }
    
    def _encode_text(self, text: str, max_len: int) -> list:
        """编码自然语言文本"""
        tokens = self.tokenizer.tokenize(text)
        ids = [1]  # <bos>
        
        for token_text, token_type, _, _ in tokens:
            if len(ids) >= max_len - 1:
                break
            tid = self.tokenizer._token_to_id(token_text, token_type)
            ids.append(tid)
        
        ids.append(2)  # <eos>
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        
        return ids


class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理"""
    
    server_instance: MotherModelServer = None
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body)
            intent = data.get('intent', '看看这段代码有没有问题')
            code = data.get('code', '')
            language = data.get('language', '')
            
            if not code:
                self._send_error(400, "Missing 'code' field")
                return
            
            result = self.server_instance.analyze(intent, code, language)
            self._send_json(200, result)
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            self._send_error(500, str(e))
    
    def do_GET(self):
        if self.path == '/health':
            self._send_json(200, {"status": "ok", "model": "mother_model"})
        elif self.path == '/stats':
            total_params = self.server_instance.model.get_total_params()
            self._send_json(200, {
                "model": "mother_model",
                "params_m": total_params / 1e6,
                "device": str(self.server_instance.device),
            })
        else:
            self._send_json(404, {"error": "not found"})
    
    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def _send_error(self, status: int, message: str):
        self._send_json(status, {"error": message})
    
    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description="母模型推理服务")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/mother_int4.pt",
                        help="模型检查点路径")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="监听地址")
    parser.add_argument("--port", type=int, default=8888,
                        help="监听端口")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="推理设备")
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({vram:.1f}GB)")
    
    # 启动服务
    server = MotherModelServer(args.checkpoint, device)
    InferenceHandler.server_instance = server
    
    httpd = HTTPServer((args.host, args.port), InferenceHandler)
    logger.info(f"Mother Model serving on http://{args.host}:{args.port}")
    logger.info(f"  POST /analyze  - Code analysis")
    logger.info(f"  GET  /health   - Health check")
    logger.info(f"  GET  /stats    - Model stats")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
