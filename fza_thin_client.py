"""
fza_thin_client.py — The Mobile Root System (Edge-Only Node)
=============================================================
A minimal FZA node that runs only the fast tiers of the inference stack:
  Stage 1: FZAReflexNode — O(1) pattern matching
  Stage 2: FZAMemoryGraph — cosine similarity over adapter bank

No GPU, no LLM, no LoRA fine-tuning. Pure CPU, ultra-low power.

This is the equivalent of the iOS- or Raspberry Pi-deployed
"edge reflex layer" from the Mycorrhizal Network architecture.
70% of daily queries are handled here without ever waking the main engine.

Usage:
  source .venv/bin/activate
  python fza_thin_client.py

For deferred heavy queries, it sends them to the main FZA engine
at http://localhost:8000 (or a configured broker URL).
"""

import json
import requests
import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from fza_reflex_node import FZAReflexNode
from fza_memory_graph import FZAMemoryGraph
from fza_event_bus import bus


BROKER_URL = "http://localhost:8001"
CORTEX_URL = "http://localhost:8000"


class FZAThinClient:
    """
    A lightweight, GPU-free FZA edge node.
    
    Layer 1 → FZAReflexNode (keyword intercept, O(1) RAM lookup)
    Layer 2 → FZAMemoryGraph (cosine-similarity associative recall)
    Layer 3 → Defer to Cortex (Mac main engine, async HTTP)
    """

    def __init__(self, profile_path: str = "user_profile.json", broker_url: str = BROKER_URL):
        self.broker_url = broker_url
        self.cortex_url = CORTEX_URL
        
        # Load user profile from disk (synced from Mac)
        self.user_profile = self._load_profile(profile_path)
        
        # Stage 1: Reflex Node
        self.reflex = FZAReflexNode()
        
        # Stage 2: Memory Graph (adapter bank loaded from disk)
        adapter_base = self.user_profile.get("_adapter_base", "./adapters")
        self.memory_graph = FZAMemoryGraph(adapter_base=adapter_base)
        
        print("⚡ [ThinClient] Edge node initialized (CPU-only, no LLM)")
        print(f"   Broker: {self.broker_url}")
        print(f"   Profile facts: {len([k for k in self.user_profile if not k.startswith('_')])}")

    def _load_profile(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def query(self, message: str) -> str:
        """
        Routes the query through the three-stage edge stack.
        Returns a response string or a deferral notice.
        """
        bus.emit("thin_client_query", {"message": message[:80]})
        
        # ── Stage 1: Jellyfish Reflex (0ms) ─────────────────────────────────
        instant = self.reflex.intercept(message, self.user_profile)
        if instant is not None:
            bus.emit("reflex_intercept", {"type": "thin_client_v6", "query": message})
            print(f"⚡ [ThinClient] Reflex hit → {instant[:60]}")
            return instant
        
        # ── Stage 2: Memory Graph Cosine Recall ─────────────────────────────
        graph_result = self._graph_recall(message)
        if graph_result:
            print(f"🧠 [ThinClient] Graph hit → {graph_result[:60]}")
            return graph_result
        
        # ── Stage 3: Defer to Cortex ─────────────────────────────────────────
        return self._defer_to_cortex(message)

    def _graph_recall(self, message: str, threshold: float = 0.75) -> str | None:
        """
        Queries the Memory Graph for a strong cosine-similarity hit.
        Returns None if no adapter is confident enough.
        """
        try:
            neighbors = self.memory_graph.expand_neighbors(message, top_k=1)
            if neighbors:
                top_id, score = neighbors[0]
                if score >= threshold:
                    meta = self.memory_graph.adapter_bank.get_meta(top_id)
                    if meta and meta.get("facts"):
                        fact_text = "; ".join(meta["facts"][:2])
                        return f"[장기기억 회상] {fact_text}"
        except Exception as e:
            print(f"⚠️ [ThinClient] Graph recall error: {e}")
        return None

    def _defer_to_cortex(self, message: str) -> str:
        """
        Forwards the query to the main FZA cortex engine.
        In a real deployment this would be async. Here it's a sync HTTP call.
        """
        bus.emit("deferred_to_cortex", {"message": message[:80]})
        print(f"🌐 [ThinClient] Cortex로 전달: {message[:50]}")
        try:
            resp = requests.post(
                f"{self.cortex_url}/chat",
                json={"message": message},
                timeout=60,
            )
            if resp.ok:
                return resp.json().get("reply", "(응답 없음)")
        except requests.RequestException as e:
            print(f"⚠️ [ThinClient] Cortex 연결 실패: {e}")
        return "[오프라인] 메인 엔진에 연결할 수 없습니다. 네트워크를 확인해주세요."

    def sync_adapters_from_broker(self):
        """Downloads new adapters from the Mycorrhizal broker."""
        try:
            resp = requests.get(f"{self.broker_url}/adapters/list", timeout=10)
            if not resp.ok:
                print(f"⚠️ [ThinClient] Broker 목록 조회 실패")
                return
            
            catalog = resp.json().get("adapters", [])
            from fza_sync_protocol import unpack_adapter
            
            synced = 0
            for entry in catalog:
                adapter_id = entry["adapter_id"]
                blob_resp = requests.get(f"{self.broker_url}/adapters/{adapter_id}", timeout=30)
                if blob_resp.ok:
                    unpack_adapter(blob_resp.json(), destination_dir="./adapters")
                    synced += 1
            
            bus.emit("adapters_synced", {"count": synced})
            print(f"📥 [ThinClient] {synced}개 어댑터 동기화 완료")
        except Exception as e:
            print(f"⚠️ [ThinClient] 동기화 실패: {e}")

    def run_interactive(self):
        """Interactive REPL for the thin client."""
        print("\n🌿 FZA Thin Client — Edge Node 대화 모드")
        print("Commands: '동기화' to sync adapters, 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "종료"):
                    break
                if user_input == "동기화":
                    self.sync_adapters_from_broker()
                    continue
                
                reply = self.query(user_input)
                print(f"Edge: {reply}\n")
                
            except KeyboardInterrupt:
                break
        
        print("\n👋 Thin Client 종료.")


if __name__ == "__main__":
    client = FZAThinClient()
    client.run_interactive()
