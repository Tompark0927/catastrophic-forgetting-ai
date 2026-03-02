"""
FZA RAG Memory — 벡터 DB 기반 의미론적 장기 기억
────────────────────────────────────────────────────
sentence-transformers + FAISS 로 수만 개의 사실을 저장하고
키워드 일치가 아닌 '의미' 기반으로 검색합니다.

user_profile (flat dict) 의 한계:
  - 항목이 많아지면 context window 초과
  - 정확한 키워드가 있어야만 검색 가능

FZAMemory 의 장점:
  - 수만 개 사실 저장 가능 (FAISS 인덱스)
  - "내 직업이 뭐야?" → "나는 개발자다" 를 자동 연결
  - 질문과 관련된 기억 top-k 개만 꺼내 context overflow 방지

의존성: pip install sentence-transformers faiss-cpu
"""
import os
import json
import numpy as np


class FZAMemory:
    EMBED_MODEL = "all-MiniLM-L6-v2"  # 384차원, 소형·고속
    DIM = 384

    def __init__(self):
        self._embed_model = None  # lazy: sentence_transformers 첫 접근 시 로드
        self._index = None        # lazy: faiss 첫 접근 시 초기화
        self._texts: list = []

    # ── lazy 프로퍼티 ──────────────────────────────────────────
    @property
    def embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.EMBED_MODEL)
        return self._embed_model

    @property
    def index(self):
        if self._index is None:
            import faiss
            # Inner Product (L2-정규화 후 = 코사인 유사도)
            self._index = faiss.IndexFlatIP(self.DIM)
        return self._index

    # ── 기억 추가 ─────────────────────────────────────────────
    def add(self, text: str):
        """텍스트를 임베딩하여 벡터 인덱스에 저장합니다."""
        emb = self.embed_model.encode([text], normalize_embeddings=True)
        self.index.add(emb.astype(np.float32))
        self._texts.append(text)
        print(f"🧠 [RAG] '{text[:40]}' 기억 추가. (총 {len(self._texts)}개)")

    # ── 의미 검색 ─────────────────────────────────────────────
    def recall(self, query: str, top_k: int = 3) -> list:
        """쿼리와 의미적으로 가장 유사한 기억을 반환합니다."""
        if self.index.ntotal == 0:
            return []
        emb = self.embed_model.encode([query], normalize_embeddings=True)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(emb.astype(np.float32), k)
        return [self._texts[i] for i in indices[0] if i >= 0]

    # ── 저장 / 불러오기 ───────────────────────────────────────
    def save(self, path="vault/rag_memory"):
        import faiss
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/texts.json", "w", encoding="utf-8") as f:
            json.dump(self._texts, f, ensure_ascii=False, indent=2)
        print(f"💾 [RAG] {len(self._texts)}개 벡터 기억 저장 완료.")

    def load(self, path="vault/rag_memory", silent=False):
        import faiss
        if not os.path.exists(f"{path}/index.faiss"):
            if not silent:
                print("❌ [RAG] 저장된 벡터 기억이 없습니다.")
            return False
        self._index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/texts.json", "r", encoding="utf-8") as f:
            self._texts = json.load(f)
        print(f"📂 [RAG] {len(self._texts)}개 벡터 기억 복구 완료.")
        return True

    def __len__(self):
        return len(self._texts)
