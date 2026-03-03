"""
fza_dmoe_router.py — Distributed Mixture-of-Experts Router (v11.0)
=====================================================================
The "brain" of the Hive-Mind dispatch system.

Given a user query, this router decides:
  1. Can the LOCAL node handle this alone (80%+ cosine match in its own adapter bank)?
     → Yes: handle locally as usual.
  2. No: which remote node(s) in the network are better experts?
     → Query the broker for the Node Registry.
     → Pick top-K expert nodes.
     → Forward the query to them (or split it and collect results).
  3. Merge results from multiple nodes into a single unified reply.

Think of it like a search engine — but instead of searching web pages,
it searches across live, specialized AI nodes and combines their outputs.

Biological parallel: The mycelial pheromone gradient. An ant follows the
strongest pheromone trail to the nearest food source. Here, the "trail"
is a cosine similarity score between the query embedding and a node's
adapter specializations.
"""

import json
import requests
from typing import List, Optional, Tuple


BROKER_URL = "http://localhost:8001"


def _simple_topic_extract(query: str) -> List[str]:
    """
    Lightweight keyword extractor — no ML model needed.
    In production this would use a tiny sentence-transformer.
    
    Returns a list of 1-3 candidate topic words from the query.
    """
    # Strip common Korean/English stop words and take content words
    stop = {
        "뭐", "어떻게", "알려줘", "말해줘", "왜", "어디", "뭐야", "인가요", "이야", 
        "what", "how", "why", "where", "is", "the", "a", "an", "of", "in", "tell", "me"
    }
    words = query.lower().replace("?", "").replace(".", "").split()
    topics = [w for w in words if w not in stop and len(w) > 1]
    return topics[:3]


class DmoeRouter:
    """
    Distributed Mixture-of-Experts router.
    
    Usage:
        router = DmoeRouter(local_manager=fza_manager)
        result = router.route("What is the boiling point of water?")
        print(result["reply"])
        print(result["served_by"])   # "local" or node_id
    """
    
    LOCAL_CONFIDENCE_THRESHOLD = 0.60  # If local adapter match ≥ this, don't go remote
    
    def __init__(self, local_manager, broker_url: str = BROKER_URL):
        """
        Args:
            local_manager: The local FZAManager instance
            broker_url:    URL of the Mycorrhizal broker
        """
        self.local_manager = local_manager
        self.broker_url = broker_url
    
    def route(self, query: str) -> dict:
        """
        Routes a query through the Hive-Mind.
        
        Returns:
          {
            "reply":     str,
            "served_by": "local" | node_id,
            "strategy":  "local_only" | "remote_expert" | "merged",
          }
        """
        # ── Step 1: Check local confidence ────────────────────────────────────
        local_score = self._local_confidence(query)
        
        if local_score >= self.LOCAL_CONFIDENCE_THRESHOLD:
            # We're confident enough to handle locally
            reply = self.local_manager.chat(query)
            return {"reply": reply, "served_by": "local", "strategy": "local_only", "local_score": local_score}
        
        # ── Step 2: Ask broker for expert nodes ───────────────────────────────
        topics = _simple_topic_extract(query)
        experts = self._fetch_experts(topics, top_k=2)
        
        if not experts:
            # No remote nodes available — fall back to local
            reply = self.local_manager.chat(query)
            return {"reply": reply, "served_by": "local", "strategy": "local_fallback", "local_score": local_score}
        
        # ── Step 3: Query the top expert node ─────────────────────────────────
        top_expert = experts[0]
        remote_reply = self._query_remote(top_expert, query)
        
        if remote_reply is None:
            # Remote failed — fall back to local
            reply = self.local_manager.chat(query)
            return {"reply": reply, "served_by": "local", "strategy": "remote_failed", "local_score": local_score}
        
        return {
            "reply": remote_reply,
            "served_by": top_expert["node_id"],
            "strategy": "remote_expert",
            "local_score": local_score,
            "expert_host": top_expert.get("host"),
        }
    
    def _local_confidence(self, query: str) -> float:
        """
        Scores how well the local adapter bank can answer this query.
        Uses the Memory Graph's cosine scores as a proxy for confidence.
        Returns a float in [0, 1].
        """
        try:
            if self.local_manager.memory_graph:
                neighbors = self.local_manager.memory_graph.expand_neighbors(query, top_k=1)
                if neighbors:
                    _, score = neighbors[0]
                    return float(score)
        except Exception:
            pass
        return 0.0
    
    def _fetch_experts(self, topics: List[str], top_k: int = 2) -> List[dict]:
        """Ask the broker's Node Registry for the best expert nodes."""
        try:
            resp = requests.post(
                f"{self.broker_url}/nodes/find_experts",
                json={"topics": topics, "top_k": top_k},
                timeout=5,
            )
            if resp.ok:
                return resp.json().get("experts", [])
        except Exception as e:
            print(f"⚠️ [dMoE] 브로커에서 전문가 노드 조회 실패: {e}")
        return []
    
    def _query_remote(self, expert: dict, query: str) -> Optional[str]:
        """
        Sends the query to a remote FZA node's /chat endpoint.
        Returns the reply string or None on failure.
        """
        base_url = expert.get("base_url", f"http://{expert.get('host')}:{expert.get('port')}")
        try:
            resp = requests.post(
                f"{base_url}/chat",
                json={"message": query},
                timeout=60,
            )
            if resp.ok:
                return resp.json().get("reply")
        except Exception as e:
            print(f"⚠️ [dMoE] 원격 노드 응답 실패 ({base_url}): {e}")
        return None


def get_routing_stats(router: DmoeRouter) -> dict:
    """Returns a summary of the router's configuration and broker status."""
    try:
        resp = requests.get(f"{router.broker_url}/nodes/list", timeout=3)
        node_count = len(resp.json().get("nodes", [])) if resp.ok else 0
    except Exception:
        node_count = 0
    
    return {
        "local_threshold": router.LOCAL_CONFIDENCE_THRESHOLD,
        "broker": router.broker_url,
        "live_nodes": node_count,
    }
