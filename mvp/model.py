"""ClawNet 编程MVP — 模型调用接口

支持两种模式:
  simulation  返回预置的模拟响应
  api         调用 DeepSeek API
"""

import json
import urllib.request
from typing import Optional

from config import MODE, DEEPSEEK_API_KEY, API_BASE, DEEPSEEK_MODEL
from child_models import ChildModel

# ─── 模拟响应库（仿真模式使用） ────────────────────────────

MOCK_RESPONSES = {
    "sql": """-- schema/users.sql
CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(64) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_username ON users(username);
""",

    "go": """package handler

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type LoginResponse struct {
	Token string `json:"token"`
}

var jwtSecret = []byte("change-me-in-production")

func LoginHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"username": req.Username,
		"exp":      time.Now().Add(24 * time.Hour).Unix(),
	})

	tokenStr, err := token.SignedString(jwtSecret)
	if err != nil {
		http.Error(w, "failed to generate token", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(LoginResponse{Token: tokenStr})
}
""",

    "react": """import React, { useState } from 'react';

interface LoginFormProps {
  onLoginSuccess: (token: string) => void;
}

export const LoginForm: React.FC<LoginFormProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) throw new Error('Login failed');

      const data = await res.json();
      onLoginSuccess(data.token);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="login-form">
      <h2>Login</h2>
      {error && <p className="error">{error}</p>}
      <input
        type="text"
        placeholder="Username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        required
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
      />
      <button type="submit" disabled={loading}>
        {loading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
};
""",
}

def call_simulation(child: ChildModel, prompt: str) -> str:
    """根据子模型的语言返回对应的模拟代码"""
    lang_key = child.language.lower()

    if "sql" in lang_key:
        return MOCK_RESPONSES["sql"]
    if "go" in lang_key:
        return MOCK_RESPONSES["go"]
    if "typescript" in lang_key or "jsx" in lang_key:
        return MOCK_RESPONSES["react"]

    return f"// {child.name} 模拟输出\n"

def call_api(child: ChildModel, prompt: str) -> str:
    """调用 DeepSeek API 获取响应"""
    if not DEEPSEEK_API_KEY:
        print(f"  ⚠ 未设置 DEEPSEEK_API_KEY，回退到仿真模式")
        return call_simulation(child, prompt)

    messages = [
        {"role": "system", "content": child.system_prompt},
        {"role": "user", "content": prompt},
    ]

    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.1,
    }).encode()

    req = urllib.request.Request(
        API_BASE,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  ⚠ API 调用失败: {e}")
        print(f"  回退到仿真模式")
        return call_simulation(child, prompt)

def call_model(child: ChildModel, prompt: str) -> str:
    """统一模型调用接口"""
    if MODE == "api":
        return call_api(child, prompt)
    return call_simulation(child, prompt)
