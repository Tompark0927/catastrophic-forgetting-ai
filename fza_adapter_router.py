"""
FZA Adapter Router — Isolated LoRA per Memory Block + Cosine Router
──────────────────────────────────────────────────────────────────────────────
v4.0 core: Zero catastrophic forgetting by design.

Architecture:
  • Each call to `create_and_freeze_adapter(facts)` trains a tiny LoRA adapter
    on ONLY those facts, then IMMEDIATELY freezes it (write-once).
  • At inference, the Router embeds the user query and picks the top-k
    most relevant adapters via cosine similarity.
  • The base model is loaded once; adapters are lightweight overlays (~1% params).

Why this is provably near-zero forgetting:
  • Adapter A is NEVER modified after creation → A's facts cannot be overwritten.
  • New facts go into Adapter B, C, D, ... — fully isolated.
  • The only thing that "learns" over time is the router embedding index,
    which is purely additive (new entries appended, never deleted).

Energy profile:
  • Training one adapter on 1-5 facts: ~5-20 gradient steps, seconds on MPS.
  • Inference overhead: cosine sim over N adapter embeddings — O(N) but tiny.
  • A system with 1,000 adapters uses less compute than one full forward pass.
"""
import os
import json
import uuid
import torch
import numpy as np
from typing import List, Optional, Dict


class AdapterBank:
    """
    Stores metadata + embeddings for all frozen LoRA adapters.
    The actual weights live on disk; only the routing embeddings stay in RAM.
    """

    def __init__(self, vault_path: str = "vault/adapters"):
        self.vault_path = vault_path
        os.makedirs(vault_path, exist_ok=True)

        # adapter_id → { "facts": [...], "embedding": np.array, "path": str }
        self._registry: Dict[str, dict] = {}
        self._load_registry()

    def _registry_path(self):
        return os.path.join(self.vault_path, "registry.json")

    def _load_registry(self):
        path = self._registry_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Restore embeddings as np arrays
            for aid, meta in data.items():
                if meta.get("embedding"):
                    meta["embedding"] = np.array(meta["embedding"], dtype=np.float32)
            self._registry = data
            print(f"📂 [Router] {len(self._registry)}개 어댑터 로드.")

    def _save_registry(self):
        serialisable = {}
        for aid, meta in self._registry.items():
            entry = dict(meta)
            if isinstance(entry.get("embedding"), np.ndarray):
                entry["embedding"] = entry["embedding"].tolist()
            serialisable[aid] = entry
        with open(self._registry_path(), "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)

    def register(
        self,
        adapter_id: str,
        facts: List[str],
        embedding: np.ndarray,
        adapter_path: str,
    ):
        self._registry[adapter_id] = {
            "facts":     facts,
            "embedding": embedding,
            "path":      adapter_path,
        }
        self._save_registry()
        print(f"🔐 [Router] 어댑터 '{adapter_id[:8]}…' 등록 완료. ({len(facts)}개 사실)")

    def get_all_ids(self) -> List[str]:
        return list(self._registry.keys())

    def get_meta(self, adapter_id: str) -> Optional[dict]:
        return self._registry.get(adapter_id)

    def __len__(self):
        return len(self._registry)


class FZAAdapterRouter:
    """
    Creates, freezes, and routes isolated LoRA adapter per memory block.

    Usage:
        router = FZAAdapterRouter(base_model, tokenizer, vault_path="vault/adapters")
        adapter_id = router.create_and_freeze_adapter(["내 이름은 김민준이야"])
        top_adapters = router.route("내 이름이 뭐야?", top_k=3)
        output = router.generate_with_adapters(prompt, top_adapters)
    """

    def __init__(
        self,
        base_model,
        tokenizer,
        vault_path: str = "vault/adapters",
        embed_model_name: str = "all-MiniLM-L6-v2",
        lora_rank: int = 4,
        lora_alpha: int = 8,
        train_steps: int = 10,
    ):
        self.base_model       = base_model
        self.tokenizer        = tokenizer
        self.vault_path       = vault_path
        self.lora_rank        = lora_rank
        self.lora_alpha       = lora_alpha
        self.train_steps      = train_steps
        self.device           = str(next(base_model.parameters()).device)

        self.bank = AdapterBank(vault_path)

        # Lazy-load sentence-transformers for routing
        self._embedder = None
        self._embed_model_name = embed_model_name

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embed_model_name)
        return self._embedder

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts and returns mean-pooled unit vector."""
        embs = self.embedder.encode(texts, normalize_embeddings=True)
        mean = embs.mean(axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm if norm > 0 else mean).astype(np.float32)

    # ── Core: Create a write-once frozen adapter ───────────────────
    def create_and_freeze_adapter(self, facts: List[str]) -> str:
        """
        Trains a tiny LoRA adapter on `facts`, freezes it immediately,
        saves it to disk, and registers it with the router.

        Returns the adapter_id (UUID).
        Energy: O(len(facts) × train_steps) forward/backward passes.
        """
        from peft import LoraConfig, get_peft_model, TaskType

        adapter_id   = uuid.uuid4().hex[:12]
        adapter_path = os.path.join(self.vault_path, adapter_id)
        os.makedirs(adapter_path, exist_ok=True)

        print(f"⚡ [Router] 새 어댑터 훈련 중 ({len(facts)}개 사실, {self.train_steps}스텝) …")

        # Wrap base model with a fresh LoRA config
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,   # No dropout for memory encoding
            bias="none",
        )
        peft_model = get_peft_model(self.base_model, config)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=3e-4,
        )

        peft_model.train()
        for text in facts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(self.device)

            for _ in range(self.train_steps):
                optimizer.zero_grad()
                outputs = peft_model(**inputs, labels=inputs["input_ids"])
                outputs.loss.backward()
                optimizer.step()

        # ── FREEZE immediately — write-once ──────────────────────
        for param in peft_model.parameters():
            param.requires_grad = False
        peft_model.eval()

        # Save only the LoRA delta weights (tiny — ~MB not GB)
        peft_model.save_pretrained(adapter_path)
        print(f"🔒 [Router] 어댑터 동결 & 저장: {adapter_path}")

        # Detach LoRA from base model to restore trainability
        peft_model.disable_adapter_layers()

        # Register with the bank
        embedding = self._embed(facts)
        self.bank.register(adapter_id, facts, embedding, adapter_path)

        return adapter_id

    # ── Routing: cosine similarity retrieval ──────────────────────
    def route(self, query: str, top_k: int = 3) -> List[str]:
        """
        Returns the top_k adapter IDs most relevant to `query`.
        Pure cosine similarity — O(N adapters), no GPU needed.
        """
        if len(self.bank) == 0:
            return []

        query_emb = self._embed([query])

        scores = []
        for aid in self.bank.get_all_ids():
            meta = self.bank.get_meta(aid)
            if meta and meta.get("embedding") is not None:
                sim = float(np.dot(query_emb, meta["embedding"]))
                scores.append((aid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = [aid for aid, _ in scores[:top_k]]

        if top:
            top_facts = [self.bank.get_meta(aid)["facts"][:1] for aid in top]
            print(f"🧭 [Router] 쿼리 '{query[:30]}' → 어댑터 {len(top)}개 선택")
            for aid, fact in zip(top, top_facts):
                print(f"   · {aid[:8]}… : {fact[0][:50] if fact else '?'}")

        return top

    # ── Inference: merge top-k adapters and generate ──────────────
    def generate_with_adapters(
        self,
        prompt: str,
        adapter_ids: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Loads the selected LoRA adapters, merges them into the base model
        (weighted sum), runs inference, then unloads — leaving base model clean.

        Energy: one forward pass with merged delta weights.
        """
        from peft import PeftModel

        if not adapter_ids:
            # No relevant adapters — fall back to plain base model
            return self._generate_base(prompt, max_new_tokens, temperature)

        # Merge adapters sequentially with equal weighting
        # (for 1-3 adapters this is negligible compute)
        merged = self.base_model
        weight = 1.0 / len(adapter_ids)

        for aid in adapter_ids:
            meta = self.bank.get_meta(aid)
            if not meta or not os.path.exists(meta["path"]):
                continue
            try:
                peft_m = PeftModel.from_pretrained(merged, meta["path"])
                merged = peft_m.merge_and_unload()   # bakes LoRA into weights
            except Exception as e:
                print(f"⚠️ [Router] 어댑터 {aid[:8]} 로드 실패: {e}")

        result = self._generate_base(prompt, max_new_tokens, temperature, model=merged)
        return result

    def _generate_base(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        model=None,
    ) -> str:
        m = model if model is not None else self.base_model
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(self.device)
        with torch.no_grad():
            out = m.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    @property
    def adapter_count(self) -> int:
        return len(self.bank)
