"""
FZA Micro Reflex Node — Middle-Tier Dynamic Sparsity (Biomimetic v8.0)
──────────────────────────────────────────────────────────────────────────────
While the v7.0 Stage 1 Reflex catches *factual* queries, it correctly escalates
complex or analytical queries to the full LLM. However, many "analytical" queries
are structurally very simple (e.g. "요약해줘" (summarize), "영어로 번역해" (translate)).

Waking up the full FZA pipeline for these trivial tasks is unnecessarily expensive:
  1. Retrieve RAG vectors
  2. Traverse Memory Graph (PageRank)
  3. Load LoRA adapters
  4. Build massive system prompt
  5. Generate

The Micro Reflex sits *between* the 0ms factual reflex and the heavy LLM pipeline.
It catches structural/functional intents and routes them to a "Sparsity Path"
— raw Mistral with a highly restricted, zero-context prompt.

Energy Profile:
  • Full Pipeline: ~4,000ms context prep + deep generation.
  • Micro Reflex:  ~50ms routing + raw model generation (zero context prep).
"""
import re
from typing import Optional, Dict
import numpy as np


class FZAMicroReflex:
    """
    Sits behind FZAReflexNode. If query requires generation but is *structurally
    simple* (translation, summarization, formatting), it returns a specific
    system prompt override that skips memory retrieval entirely.
    """

    # ── Structural Intent Banks ─────────────────────────────────────────
    INTENTS = {
        "summarize": {
            "patterns": [
                r"(요약|간단히\s*설명|정리|줄거리|줄여서|summarize|summary|tl;dr|tldr)",
                r"(핵심만|짧게|요점만)",
            ],
            "prompt": "You are a precise summarization assistant. Summarize the user's input concisely. Do not add outside information."
        },
        "translate": {
            "patterns": [
                r"(번역|통역|translate|영어로|한국어로|일본어로|중국어로)",
                r"(뜻이\s*뭐야|무슨\s*뜻)",
            ],
            "prompt": "You are an expert translator. Translate the user's text accurately to the target language implied by the context. Only output the translation."
        },
        "code": {
            "patterns": [
                r"(코드|파이썬|자바스크립트|스크립트|함수|클래스\s*짜줘|code|python|js|react)",
                r"(어떻게\s*구현|프로그래밍)",
            ],
            "prompt": "You are an expert software engineer. Provide only complete, working code snippets wrapped in markdown block backticks. Keep explanations extremely brief."
        },
        "format": {
            "patterns": [
                r"(표로\s*만들어|목록으로|포맷|정렬|형식|단락으로\s*나눠|format|table|list)",
                r"(보기\s*좋게)",
            ],
            "prompt": "You are a data formatting assistant. Reformat the user's input strictly according to their requested structure (e.g. table, bulleted list)."
        }
    }

    def __init__(self, confidence_threshold: float = 0.65, embed_model: str = "all-MiniLM-L6-v2"):
        self.threshold = confidence_threshold
        self._embedder = None
        self._embed_model = embed_model
        
        # Compiled regexes for ultra-fast Stage 1 routing
        self._regexes = {
            intent: re.compile("|".join(data["patterns"]), re.IGNORECASE)
            for intent, data in self.INTENTS.items()
        }

        # Stats
        self.total_intercepted = 0
        self.total_escalated = 0

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._embed_model)
            except ImportError:
                print("⚠️ [MicroReflex] sentence-transformers not available, using pure regex.")
        return self._embedder

    def _embed(self, texts: list) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True)

    def intercept(self, query: str) -> Optional[Dict[str, str]]:
        """
        Analyzes if the query is a purely structural/functional task.
        
        Returns:
            Dict containing {"intent": str, "system_prompt": str} if intercepted.
            None if the query is complex and needs the full FZA memory pipeline.
        """
        # Step 1: Ultra-fast regex scan
        q_norm = query.lower().strip()
        for intent, regex in self._regexes.items():
            if regex.search(q_norm):
                self.total_intercepted += 1
                return {"intent": intent, "system_prompt": self.INTENTS[intent]["prompt"]}

        # Step 2: Fall back to semantic embedding if active (optional stage)
        # For v8.0, we rely strictly on regex to guarantee ~0ms routing.
        
        self.total_escalated += 1
        return None

    def print_stats(self):
        total = self.total_intercepted + self.total_escalated
        rate = (self.total_intercepted / total) * 100 if total > 0 else 0
        print(f"🧠 [MicroReflex] 통계:")
        print(f"   총 심화 쿼리:  {total}개")
        print(f"   구조적 처리:   {self.total_intercepted}개 ({rate:.0f}%) -> RAG/Graph 무시")
        print(f"   풀 파이프라인: {self.total_escalated}개 ({100-rate:.0f}%)")
