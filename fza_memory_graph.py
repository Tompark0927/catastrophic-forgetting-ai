"""
FZA Memory Graph — Associative Memory over the Adapter Bank
──────────────────────────────────────────────────────────────────────────────
Every time a new isolated LoRA adapter is registered (via FZAAdapterRouter),
this graph is automatically updated. Adapters become nodes. Semantic similarity
between their stored facts becomes weighted edges.

The result is an EMERGENT ASSOCIATIVE MEMORY — the system doesn't just retrieve
one relevant adapter; it traverses the graph and pulls in semantically connected
neighbors. This mirrors the human associative recall mechanism:

  Human: "Where do I work?" → brain recalls "Google" → also recalls "John's boss is Sarah" → recalls "I commute 45 min"
  FZA:   Query → Router [Adapter_A: job] → Graph traversal → [Adapter_B: commute, Adapter_C: colleague]
                   → all three contexts merged into the response

This is not just retrieval. It is emergent contextual intelligence.

Key design decisions:
  • Pure numpy/scipy adjacency matrix — zero new dependencies.
  • Edges are computed lazily (only when traversal is needed).
  • Graph is stored as a dense float32 similarity matrix (symmetric, diagonal=1.0).
  • Traversal uses personalized PageRank via power iteration — stable, fast, elegant.
  • Supports adding nodes incrementally without full recomputation.
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional


class FZAMemoryGraph:
    """
    A dynamically growing associative memory graph over LoRA adapters.

    Nodes  = adapter IDs (each holding a permanently frozen semantic memory block).
    Edges  = cosine similarity between the mean-pooled embeddings of their fact sets.
    Weight = float in [0, 1] where 1.0 = identical meaning, 0.0 = completely unrelated.

    Graph is used at inference time to expand the router's top-k selection
    into a semantically enriched neighborhood.
    """

    def __init__(
        self,
        vault_path: str = "vault/memory_graph",
        similarity_threshold: float = 0.40,
        max_neighbors: int = 3,
    ):
        """
        Args:
            vault_path:            Directory to persist the graph.
            similarity_threshold:  Edges below this weight are pruned (noise gates).
            max_neighbors:         Max graph-expanded adapters per hop.
        """
        self.vault_path          = vault_path
        self.similarity_threshold = similarity_threshold
        self.max_neighbors       = max_neighbors

        os.makedirs(vault_path, exist_ok=True)

        # Ordered list of adapter IDs (index → node)
        self._nodes: List[str]    = []
        # Dense adjacency matrix — shape (N, N), float32
        self._adj: np.ndarray     = np.zeros((0, 0), dtype=np.float32)
        # adapter_id → mean embedding (l2-normalized)
        self._embeddings: Dict[str, np.ndarray] = {}
        # adapter_id → list of fact strings
        self._facts: Dict[str, List[str]] = {}

        self._load()

    # ─── Persistence ────────────────────────────────────────────────
    def _graph_path(self)  -> str: return os.path.join(self.vault_path, "graph.npz")
    def _meta_path(self)   -> str: return os.path.join(self.vault_path, "meta.json")

    def _save(self):
        np.savez_compressed(
            self._graph_path(),
            adj=self._adj,
            embeddings=np.vstack(list(self._embeddings.values())) if self._embeddings else np.zeros((0,)),
        )
        meta = {
            "nodes":  self._nodes,
            "facts":  self._facts,
            "embed_keys": list(self._embeddings.keys()),
        }
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load(self):
        if not os.path.exists(self._meta_path()):
            return
        try:
            with open(self._meta_path(), "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._nodes = meta.get("nodes", [])
            self._facts = meta.get("facts", {})
            embed_keys  = meta.get("embed_keys", [])

            data = np.load(self._graph_path(), allow_pickle=False)
            self._adj   = data["adj"]
            emb_mat     = data["embeddings"]
            if emb_mat.ndim == 2 and len(embed_keys) == emb_mat.shape[0]:
                self._embeddings = {k: emb_mat[i] for i, k in enumerate(embed_keys)}

            if len(self._nodes):
                print(f"🕸️  [MemoryGraph] 복구: {len(self._nodes)}개 노드, {self._edge_count()}개 엣지.")
        except Exception as e:
            print(f"⚠️  [MemoryGraph] 복구 실패 — {e}. 새 그래프로 시작.")

    def _edge_count(self) -> int:
        if self._adj.size == 0:
            return 0
        above_thresh = self._adj > self.similarity_threshold
        np.fill_diagonal(above_thresh, False)
        return int(above_thresh.sum()) // 2

    # ─── Node Registration ───────────────────────────────────────────
    def register_adapter(
        self,
        adapter_id: str,
        facts: List[str],
        embedding: np.ndarray,
    ):
        """
        Add a new adapter node to the graph and compute its edges to all
        existing nodes via cosine similarity.

        Called automatically by FZAAdapterRouter.create_and_freeze_adapter().
        """
        if adapter_id in self._embeddings:
            return  # Already registered

        # Normalise embedding
        norm = np.linalg.norm(embedding)
        emb  = (embedding / norm if norm > 1e-8 else embedding).astype(np.float32)

        n = len(self._nodes)

        # Grow adjacency matrix
        new_adj = np.zeros((n + 1, n + 1), dtype=np.float32)
        if n > 0:
            new_adj[:n, :n] = self._adj
            # Compute similarity to all existing nodes
            for i, existing_id in enumerate(self._nodes):
                sim = float(np.dot(emb, self._embeddings[existing_id]))
                new_adj[i, n] = sim
                new_adj[n, i] = sim
        new_adj[n, n] = 1.0  # Self-similarity is always 1

        self._adj         = new_adj
        self._nodes.append(adapter_id)
        self._embeddings[adapter_id] = emb
        self._facts[adapter_id]      = facts

        above  = int((new_adj[n, :n] > self.similarity_threshold).sum())
        print(f"🕸️  [MemoryGraph] 노드 추가: '{adapter_id[:8]}…' → {above}개 엣지 생성. "
              f"(총 {len(self._nodes)}노드 / {self._edge_count()}엣지)")
        self._save()

    # ─── Core: Graph-expanded neighborhood retrieval ─────────────────
    def expand_neighbors(
        self,
        seed_adapter_ids: List[str],
        top_k: int = None,
    ) -> List[Tuple[str, float]]:
        """
        Given a list of seed adapters (from the Router), traverse the graph
        and return an expanded set of semantically connected neighbors.

        Uses personalized PageRank (power iteration) with seeds as
        the restart distribution. Biases strongly towards highly-related
        adapters while respecting semantic distance.

        Returns:
            List of (adapter_id, relevance_score) tuples, sorted descending.
            Seed adapters are included with their PageRank score.
        """
        if not self._nodes or not seed_adapter_ids:
            return [(aid, 1.0) for aid in seed_adapter_ids]

        top_k = top_k or (len(seed_adapter_ids) + self.max_neighbors)
        n     = len(self._nodes)
        idx   = {aid: i for i, aid in enumerate(self._nodes)}

        # Build row-normalised transition matrix (only above-threshold edges)
        M = self._adj.copy()
        M[M < self.similarity_threshold] = 0.0
        np.fill_diagonal(M, 0.0)
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M /= row_sums

        # Personalisation vector: seeds get equal mass
        p = np.zeros(n, dtype=np.float32)
        valid_seeds = [s for s in seed_adapter_ids if s in idx]
        if not valid_seeds:
            return [(aid, 1.0) for aid in seed_adapter_ids]
        for s in valid_seeds:
            p[idx[s]] = 1.0 / len(valid_seeds)

        # Power iteration (personalised PageRank)
        alpha  = 0.85   # damping (follow edges 85%, restart 15%)
        r      = p.copy()
        for _ in range(30):            # 30 iterations is overkill; converges in ~10
            r_new = alpha * (M.T @ r) + (1 - alpha) * p
            if np.linalg.norm(r_new - r) < 1e-6:
                break
            r = r_new

        # Collect top-k by PageRank score
        scored = [(self._nodes[i], float(r[i])) for i in range(n) if r[i] > 1e-5]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ─── Query-centric semantic probe ────────────────────────────────
    def query_neighbors(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Direct query→graph matching: embed a query and find the most
        semantically connected adapter nodes (bypasses Router entirely).
        Used as a fallback when the Router returns 0 adapters.
        """
        if not self._nodes:
            return []
        norm = np.linalg.norm(query_embedding)
        q    = (query_embedding / norm if norm > 1e-8 else query_embedding).astype(np.float32)
        scores = []
        for aid, emb in self._embeddings.items():
            sim = float(np.dot(q, emb))
            if sim > self.similarity_threshold:
                scores.append((aid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ─── Introspection ───────────────────────────────────────────────
    def strongest_connections(self, adapter_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Returns the top-k strongest neighbors of a given adapter."""
        if adapter_id not in self._embeddings or len(self._nodes) < 2:
            return []
        idx_a = self._nodes.index(adapter_id)
        row   = self._adj[idx_a].copy()
        row[idx_a] = 0.0  # exclude self
        top_idxs   = np.argsort(row)[::-1][:top_k]
        return [(self._nodes[i], float(row[i])) for i in top_idxs if row[i] > self.similarity_threshold]

    def print_graph_summary(self):
        print(f"🕸️  [MemoryGraph] 노드: {len(self._nodes)}개 | 엣지 (≥{self.similarity_threshold:.2f}): {self._edge_count()}개")
        for aid in self._nodes[:8]:  # Show first 8
            conns = self.strongest_connections(aid, top_k=2)
            friends = ", ".join(f"{b[:6]}…({s:.2f})" for b, s in conns) or "없음"
            fact_preview = self._facts.get(aid, ["?"])[0][:40]
            print(f"   [{aid[:8]}…] '{fact_preview}' → [{friends}]")

    @property
    def node_count(self) -> int: return len(self._nodes)
    @property
    def edge_count(self) -> int: return self._edge_count()
