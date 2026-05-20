Title: Show HN: ClawNet – A P2P compute network where every job is also a witness

Every browser extension that runs a compute task also signs the result and broadcasts it to the network. The result: a distributed ledger of "what was observed by whom at what time." If someone later claims a webpage said X at time T, you can check the network's collective observation log. If no node saw it, the claim is suspect.

I call this "negative existence proof" – and it's the thing I can't stop thinking about.

The project is called ClawNet (https://github.com/asteroidea95/ClawNet). It's a pure P2P network for pooling idle devices (phone, laptop, browser) into a global compute pool. The architecture has three layers:

- **Compute layer** – browser nodes run AI tasks (inference, DOM search, image gen) via WebGPU
- **Anchor layer** – every job output is signed and gossiped to all peers, forming a tamper-evident observation chain (Ed25519 + CRDT, no blockchain)
- **.soul layer** – a file format that can only be moved, not copied (ownership chain tracked in the network's asset registry; copies without a valid chain are rejected by the protocol)

The .soul format is the part that might sound impossible. It doesn't prevent copying at the byte level – it just makes copies socially worthless. The network only recognizes the copy held by the current owner on the ownership chain. A byte-identical duplicate without a valid provenance chain gets rejected by every node in the network.

I pushed the first code yesterday (Rust core + Chrome extension + signaling server). 0 stars, 0 forks, completely fresh.

What I'd love feedback on:

1. Is "distributed evidence anchoring" a problem worth solving, or am I over-indexing on a niche fear?
2. The .soul concept – pure protocol-enforced uniqueness without hardware roots of trust. Viable long-term, or does it need TEE/Secure Enclave to be taken seriously?
3. Browser-first: is WebGPU via extension enough to attract early contributors, or should I prioritize the Rust CLI node first?
4. If you've worked on distributed systems: what's the most painful part I'm not seeing yet?

https://github.com/asteroidea95/ClawNet

(Full disclosure: this is day 1. The code compiles but hasn't run a real P2P exchange yet. Looking for sanity checks more than stars.)
