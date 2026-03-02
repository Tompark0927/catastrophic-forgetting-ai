# Catastrophic Forgetting AI (FZA System)

> *"세상에 없던 망각 없는 AI" — An AI that learns continuously without catastrophic forgetting.*

[![Status: Conceptual Prototype](https://img.shields.io/badge/Status-Conceptual_Prototype_v2.0-yellow.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Catastrophic Forgetting is the phenomenon where Artificial Neural Networks completely and abruptly forget previously learned information upon learning new information. This repository contains the **FZA (Flexible Zone Architecture) System v2.0**, an experimental framework designed to mitigate this fundamental roadblock on the path to AGI.

## 🧠 What is FZA System v2.0?

The current implementation is a conceptual prototyping framework that attempts to decouple memory zones:
1. **Root Zone (Permanent Knowledge):** Core, foundational parameters that are locked or highly preserved.
2. **Trunk Zone (Learned Skills):** Optional mid-level adaptations.
3. **Leaf Zone (Temporary Ephemera):** Highly volatile conversational context that can be "flushed" to discard short-term memory without poisoning long-term knowledge.

Currently, FZA v2.0 implements this concept via:
- **API RAG Wrapping:** Integrating Anthropic's Claude API with a local FAISS-based vector database (Vector Memory).
- **Auto-Memory Extraction:** Utilizing Small Language Models (SLMs like Claude Haiku) to automatically extract, structure, and categorize user facts on-the-fly.
- **Mock Zonal Locking:** Prototyping PyTorch gradients (`requires_grad=False`) on simple `nn.Linear` layers to simulate structural zone locking before scaling up.

### Core Architecture Components
- `fza_llm_bridge.py`: The orchestrator that integrates structural Root facts, RAG vector memory (Stage 3), and Auto-Extraction (Stage 4) via Anthropic.
- `main_fza_system.py`: The primary CLI manager for user interaction, memory testing, and mock zone training.
- `fza_storage_v2.py`: Utilities for locking, exporting, and fusing neural network block weights.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `pip install torch sentence-transformers faiss-cpu anthropic pypdf`
- Ensure you have an Anthropic API Key set up in your environment.

### Run
```bash
python main_fza_system.py
```

Basic CLI Commands:
- `물어봐 [질문]`: Ask a question directly to the AI (automatically injects RAG + Permanent Memory).
- `내 이름은 000이야`: The system will automatically parse and save this as a permanent Root Fact.
- `평생 기억해`: Locks the root zone and saves the current permanent knowledge block offline (`vault/`).
- `다 지워`: Flushes the "Leaf" volatile memory, clearing out context without losing permanent facts.
- `기억 검색 [쿼리]`: Performs a semantic search on previously flushed/saved vector memories.

## 🔭 The AGI Roadmap: Journey to v3.0 Native Local

The hard truth about catastrophic forgetting is that **prompt engineering and RAG are not fundamental cures; they are band-aids.** FZA v2.0 proves the *concept* of multi-zone memory separation, but to genuinely surprise the AI research community, the system must evolve from an API wrapper into a natively continuous-learning Transformer architecture. 

### The Native Architecture Pivot (Coming Soon)
1. **Fully Native `transformers` Integration:** Swap the Anthropic reasoning engine for a local open-weights model (e.g., Llama 3 8B / Mistral v0.3) running on MPS (Apple Silicon) or CUDA. 
2. **Elastic Weight Consolidation (EWC) / Dynamic LoRA:** Implement algorithmic penalties on critical weights to protect "Root" knowledge while allowing "Leaf" knowledge to update the model parameters natively in real-time.
3. **GraphRAG SLM Implementation:** Replace simple substring regex with an autonomous local Knowledge Graph builder. 
4. **Frozen Attention Hooks:** Directly manipulate the Attention/MLP blocks inside the LLaMA architecture, freezing the first $N$ layers as the core reasoning root, and dynamically fine-tuning the last $N$ layers during user conversations.

---
*Created by [Tompark0927](https://github.com/Tompark0927). Aiming for an AI that never forgets.*
