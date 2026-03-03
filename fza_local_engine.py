"""
FZA Local Engine — Stage 1·2·3·4 with local Mistral/Llama via HuggingFace
──────────────────────────────────────────────────────────────────────────
Drop-in replacement for fza_llm_bridge.py that uses a local model instead
of the Anthropic API.  All public methods are identical so FZAManager works
with **zero** changes when you pass  use_local=True.

Supported devices (auto-detected):
  • MPS  — Apple Silicon (M1/M2/M3)
  • CUDA — NVIDIA GPU
  • CPU  — fallback (slow for 7B models; use TinyLlama for CPU testing)

Recommended models:
  • Fast / low-RAM  → "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   (~600 MB)
  • Full quality    → "mistralai/Mistral-7B-Instruct-v0.3"     (~14 GB FP16)

Usage:
    engine = FZALocalEngine("mistralai/Mistral-7B-Instruct-v0.3")
    reply  = engine.chat("내 이름을 기억하고 있어?")
"""
import json
import os
import re
import torch
from datetime import datetime


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class FZALocalEngine:
    """
    Mirrors the public API of FZALLMBridge so FZAManager can swap engines
    transparently.  Pass this to FZAManager(bridge=...) or wire it directly.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        math_engine=None,
        memory=None,
        vault_path: str = "vault",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name     = model_name
        self.math_engine    = math_engine
        self.memory         = memory          # FZAMemory (RAG) or None
        self.vault_path     = vault_path
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature

        self.user_profile        = {}
        self.conversation_history = []

        # ── Device ────────────────────────────────────────────────
        self.device = _detect_device()
        print(f"🖥  [LocalEngine] Device: {self.device.upper()}")

        # ── Tokenizer ─────────────────────────────────────────────
        print(f"⬇️  [LocalEngine] Loading tokenizer for '{model_name}' …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Model ─────────────────────────────────────────────────
        print(f"⬇️  [LocalEngine] Loading model (this may take a while) …")
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device if self.device != "mps" else None,
            low_cpu_mem_usage=True,
        )
        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"✅ [LocalEngine] '{model_name}' ready on {self.device.upper()}.")

        # ── Phase 4: v10.0 Neuromorphic Modules ─────────────────────────────
        # 1. Native Hebbian Fast-Weights in attention blocks
        try:
            from fza_attention_patch import FZAAttentionPatcher
            self.attention_patcher = FZAAttentionPatcher(self.model, patch_layers=[0, 1, 2])
            n_patched = self.attention_patcher.apply()
            print(f"🧬 [LocalEngine] {n_patched}개 어텍션 블록에 Hebbian Fast-Weight 주입 완료")
        except Exception as e:
            self.attention_patcher = None
            print(f"⚠️ [LocalEngine] AttentionPatch 실패 (무시): {e}")

        # 2. Infinite Context Manager (rolling compression)
        try:
            from fza_infinite_context import InfiniteContextManager
            self.infinite_ctx = InfiniteContextManager(
                tokenizer=self.tokenizer, manager=None, max_tokens=3200
            )
            print(f"♾️ [LocalEngine] Infinite Context Manager 초기화 (유효: {self.infinite_ctx.max_tokens}토큰)")
        except Exception as e:
            self.infinite_ctx = None
            print(f"⚠️ [LocalEngine] InfiniteCtx 실패 (무시): {e}")

        # 3. MetaCognition Engine (self-modification)
        try:
            from fza_meta_cognition import MetaCognitionEngine
            self.meta_cognition = MetaCognitionEngine(manager=None)
            print(f"🧠 [LocalEngine] MetaCognition Engine 초기화 완료")
        except Exception as e:
            self.meta_cognition = None
            print(f"⚠️ [LocalEngine] MetaCog 실패 (무시): {e}")

    # ── Internal: build the system + user context prompt ──────────
    def _build_chat_prompt(self, user_message: str) -> str:
        """
        Constructs a Mistral-style [INST]...[/INST] prompt that includes:
          • Root facts (user profile)
          • General memories (last 30)
          • RAG recalled memories (top-3 semantic)
          • Math formulas (if any)
        """
        context_parts = []

        structured = {k: v for k, v in self.user_profile.items()
                      if not k.startswith("_")}
        if structured:
            context_parts.append("[사용자 프로필]")
            for k, v in structured.items():
                context_parts.append(f"- {k}: {v}")

        memories = self.user_profile.get("_memories", [])
        if memories:
            context_parts.append("\n[기억하고 있는 사실들]")
            for mem in memories[-30:]:
                context_parts.append(f"- {mem}")

        if self.memory and user_message:
            recalled = self.memory.recall(user_message, top_k=3)
            if recalled:
                context_parts.append("\n[지금 대화와 관련된 기억]")
                for mem in recalled:
                    context_parts.append(f"- {mem}")

        if self.math_engine and self.math_engine.math_vault:
            context_parts.append("\n[수식 — 100% 정확도 보장]")
            for name, formula in self.math_engine.math_vault.items():
                context_parts.append(f"- {name}: {formula}")

        context_block = "\n".join(context_parts)
        system = (
            "당신은 사용자를 오래 알아온 개인 AI 어시스턴트입니다.\n"
            "아래 정보를 바탕으로 자연스럽게 대화하세요.\n\n"
            + context_block
        )

        # Mistral instruct format
        prompt = f"<s>[INST] {system}\n\n{user_message} [/INST]"
        return prompt

    # ── Internal: run inference ────────────────────────────────────
    def _generate(self, prompt: str) -> str:
        from transformers import TextIteratorStreamer
        from threading import Thread
        from fza_event_bus import bus
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Run generation in a background thread so we can yield tokens
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for token in streamer:
            generated_text += token
            # Emit raw token for real-time WebSocket streaming
            bus.emit("token", {"text": token})
            print(token, end="", flush=True)

        thread.join()

        # ── Phase 4: MetaCognition post-processing ────────────────────────
        if getattr(self, 'meta_cognition', None):
            clean_output, directives = self.meta_cognition.process(generated_text)
            if directives:
                from fza_event_bus import bus as _bus
                _bus.emit("meta_cognition", {"directives": directives})
            return clean_output

        return generated_text
        print() # Newline after generation completes

        return generated_text.strip()

    # ── Public: chat (mirrors FZALLMBridge.chat) ──────────────────
    def chat(self, user_message: str) -> str:
        # Stage 1 — math override
        if self.math_engine:
            for name, formula in self.math_engine.math_vault.items():
                if name in user_message:
                    return f"📐 {name}: {formula}"

        self.conversation_history.append({"role": "user", "content": user_message})
        prompt = self._build_chat_prompt(user_message)
        
        # v8.0: PageRank LoRA Interpolation Routing
        # If the manager attached a router to us, use it.
        router = getattr(self, "router", None)
        if router:
            # 1. Expand graph to get neighbors and PageRank scores
            if hasattr(router, "memory_graph") and router.memory_graph:
                # Find direct cosine matches first to use as seeds
                seeds = router.route(user_message, top_k=2)
                if seeds:
                    neighbors = router.memory_graph.expand_neighbors(seeds, top_k=3)
                    adapter_ids = [n[0] for n in neighbors]
                    adapter_weights = [n[1] for n in neighbors]
                else:
                    adapter_ids = []
                    adapter_weights = []
            else:
                # Fallback to simple cosine routing if graph is disabled
                adapter_ids = router.route(user_message, top_k=3)
                adapter_weights = [1.0 / len(adapter_ids)] * len(adapter_ids) if adapter_ids else []

            reply = router.generate_with_adapters(
                prompt=prompt,
                adapter_ids=adapter_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                adapter_weights=adapter_weights,
            )
        else:
            # Plain base model generation
            reply = self._generate(prompt)
            
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    # ── Public: auto-extract memory (Stage 4 — local) ─────────────
    def auto_extract_memory(self, user_msg: str, ai_response: str) -> list:
        """
        Uses the same local model (with a lightweight extraction prompt)
        to find memorable facts — no external API call.
        """
        extraction_prompt = (
            "<s>[INST] 다음 대화에서 사용자에 대해 기억할 만한 구체적 사실만 추출해줘.\n"
            "이름, 직업, 나이, 사는 곳, 취미, 목표 등. 일반 질문/상식은 제외.\n"
            "없으면 빈 배열 []만 반환. JSON 배열 형식으로만 답해.\n\n"
            f"사용자: {user_msg}\nAI: {ai_response[:300]} [/INST]"
        )
        raw = self._generate(extraction_prompt)
        try:
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start >= 0 and end > start:
                facts = json.loads(raw[start:end])
                return facts if isinstance(facts, list) else []
        except Exception:
            pass
        return []

    # ── Profile helpers (identical to FZALLMBridge) ───────────────
    def set_user_fact(self, key: str, value: str):
        self.user_profile[key] = value
        print(f"🧠 [루트 지식] '{key}: {value}' 저장됨.")

    @staticmethod
    def _categorize(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["나이","생일","이름","성별"]): return "신상"
        if any(k in t for k in ["직업","일","회사","학교","전공"]): return "직업"
        if any(k in t for k in ["가족","친구","연인","부모"]): return "관계"
        if any(k in t for k in ["목표","꿈","계획","원해"]): return "목표"
        if any(k in t for k in ["건강","운동","병","다이어트"]): return "건강"
        if any(k in t for k in ["취미","즐기","관심","좋아"]): return "취미"
        if any(k in t for k in ["사는","살고","거주"]): return "장소"
        return "기타"

    def add_memory(self, text: str):
        mems  = self.user_profile.setdefault("_memories", [])
        dates = self.user_profile.setdefault("_memory_dates", [])
        cats  = self.user_profile.setdefault("_memory_cats", [])
        if text not in mems:
            mems.append(text)
            dates.append(datetime.now().isoformat(timespec="seconds"))
            cats.append(self._categorize(text))
            print(f"💡 [일반 기억] '{text[:50]}' 저장됨. (총 {len(mems)}개)")

    def save_profile(self):
        os.makedirs(self.vault_path, exist_ok=True)
        path = os.path.join(self.vault_path, "user_profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.user_profile, f, ensure_ascii=False, indent=2)
        print(f"💾 [프로필 저장] {path}")

    def load_profile(self, silent=False):
        path = os.path.join(self.vault_path, "user_profile.json")
        if not os.path.exists(path):
            if not silent:
                print("❌ [프로필] 저장된 프로필이 없습니다.")
            return
        with open(path, "r", encoding="utf-8") as f:
            self.user_profile = json.load(f)
        structured = {k: v for k, v in self.user_profile.items() if not k.startswith("_")}
        memories   = self.user_profile.get("_memories", [])
        print(f"📂 [프로필 복구] 구조화 {len(structured)}개 / 일반 기억 {len(memories)}개 로드 완료.")

    def flush_conversation(self):
        self.conversation_history = []
        print("🍂 [초기화] 대화 기록을 비웠습니다. 기억은 유지됩니다.")

    def smart_merge_memories(self, new_facts: list) -> tuple:
        """Simple add-only merge when running locally (no haiku cross-check)."""
        added = []
        for fact in new_facts:
            mems = self.user_profile.get("_memories", [])
            if fact not in mems:
                self.add_memory(fact)
                added.append(fact)
        return added, []

    def print_status(self):
        structured = {k: v for k, v in self.user_profile.items() if not k.startswith("_")}
        print(f"🖥  [LocalEngine] Model : {self.model_name}")
        print(f"🖥  [LocalEngine] Device: {self.device.upper()}")
        print(f"👤 루트 프로필: {len(structured)}개 항목")
        memories = self.user_profile.get("_memories", [])
        print(f"💡 일반 기억: {len(memories)}개")
        rag_count = len(self.memory) if self.memory else 0
        print(f"🧠 RAG 벡터 기억: {rag_count}개")
        print(f"💬 현재 대화 턴: {len(self.conversation_history) // 2}회")

    # ── Expose model for EWC / LoRA ───────────────────────────────
    @property
    def raw_model(self) -> torch.nn.Module:
        """Returns the underlying nn.Module for EWC and LoRA access."""
        return self.model
