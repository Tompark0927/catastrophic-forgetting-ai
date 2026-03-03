"""
fza_distributed_inference.py — Hive-Mind Query Relay (v11.0)
=============================================================
Handles the actual network mechanics of distributed inference:
splitting a query across multiple expert nodes and merging results.

Three strategies are supported:

1. SINGLE EXPERT (default)
   Route entire query to one expert node. Simple and low-latency.
   Best for: specific domain questions where one node is clearly dominant.

2. PARALLEL ENSEMBLE
   Send the query to top-K nodes simultaneously. Combine all replies
   by picking the longest/most coherent one or by weighted averaging.
   Best for: ambiguous queries where the best expert is uncertain.

3. CHAIN-OF-THOUGHT SPLIT (advanced)
   Node 1 generates a "thinking frame" (e.g., problem decomposition).
   Node 2 receives that frame plus the original query and generates the final answer.
   Best for: complex multi-step reasoning tasks.

Usage:
    from fza_distributed_inference import DistributedInferenceEngine
    die = DistributedInferenceEngine(broker_url="http://localhost:8001")
    result = die.query_ensemble("What is the boiling point of water?", top_k=2)
"""

import concurrent.futures
import time
import requests
from typing import List, Optional


BROKER_URL = "http://localhost:8001"


class RemoteQueryResult:
    """Container for a remote query result."""
    
    def __init__(self, node_id: str, host: str, port: int, reply: Optional[str], latency_ms: float, error: Optional[str] = None):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.reply = reply
        self.latency_ms = latency_ms
        self.error = error
        self.succeeded = reply is not None
    
    def __repr__(self):
        status = f"✅ {len(self.reply)} chars" if self.succeeded else f"❌ {self.error}"
        return f"<RemoteResult node={self.node_id[:8]} latency={self.latency_ms:.0f}ms {status}>"


def _query_single_node(node: dict, query: str, timeout_s: int = 30) -> RemoteQueryResult:
    """Sends a query to one remote FZA node and returns the result."""
    node_id = node.get("node_id", "unknown")
    host = node.get("host", "localhost")
    port = node.get("port", 8000)
    base_url = node.get("base_url", f"http://{host}:{port}")
    
    t0 = time.time()
    try:
        resp = requests.post(
            f"{base_url}/chat",
            json={"message": query},
            timeout=timeout_s,
        )
        latency = (time.time() - t0) * 1000
        if resp.ok:
            reply = resp.json().get("reply", "")
            return RemoteQueryResult(node_id, host, port, reply, latency)
        else:
            return RemoteQueryResult(node_id, host, port, None, latency, error=f"HTTP {resp.status_code}")
    except Exception as e:
        latency = (time.time() - t0) * 1000
        return RemoteQueryResult(node_id, host, port, None, latency, error=str(e))


class DistributedInferenceEngine:
    """
    Orchestrates distributed inference across multiple FZA nodes.
    """
    
    def __init__(self, broker_url: str = BROKER_URL):
        self.broker_url = broker_url
        self.total_distributed_queries = 0
        self.total_saved_ms = 0.0   # Estimated latency vs single-node baseline
    
    def get_live_nodes(self) -> List[dict]:
        """Fetches the current live node list from the broker."""
        try:
            resp = requests.get(f"{self.broker_url}/nodes/list", timeout=5)
            if resp.ok:
                return resp.json().get("nodes", [])
        except Exception as e:
            print(f"⚠️ [DistInfer] 브로커에서 노드 목록 가져오기 실패: {e}")
        return []
    
    def query_single_expert(self, query: str, topics: List[str] = None) -> Optional[RemoteQueryResult]:
        """
        Finds the single best expert node for the query and queries it.
        Returns None if no live nodes are available.
        """
        try:
            resp = requests.post(
                f"{self.broker_url}/nodes/find_experts",
                json={"topics": topics or [], "top_k": 1},
                timeout=5,
            )
            if not resp.ok:
                return None
            experts = resp.json().get("experts", [])
            if not experts:
                return None
            
            result = _query_single_node(experts[0], query)
            self.total_distributed_queries += 1
            print(f"🌐 [DistInfer] 단일 전문가 위임: {result}")
            return result
        except Exception as e:
            print(f"⚠️ [DistInfer] 단일 전문가 조회 실패: {e}")
            return None
    
    def query_ensemble(self, query: str, top_k: int = 2, timeout_s: int = 30) -> dict:
        """
        Queries top-K expert nodes in PARALLEL and merges results.
        
        Merge strategy: pick the longest coherent reply.
        (In production, this could use a small judge model to select the best.)
        
        Returns:
          {
            "merged_reply": str,
            "results":      [RemoteQueryResult],
            "strategy":     "ensemble",
          }
        """
        nodes = self.get_live_nodes()
        if not nodes:
            return {"merged_reply": None, "results": [], "strategy": "no_nodes"}
        
        target_nodes = nodes[:top_k]
        print(f"🌐 [DistInfer] 앙상블 {len(target_nodes)}개 노드에 병렬 쿼리 시작...")
        
        # Fire all requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=top_k) as pool:
            futures = {pool.submit(_query_single_node, node, query, timeout_s): node for node in target_nodes}
            results = []
            for fut in concurrent.futures.as_completed(futures, timeout=timeout_s + 2):
                try:
                    results.append(fut.result())
                except Exception as e:
                    pass
        
        # Merge: pick the best successful reply (longest non-empty one)
        successful = [r for r in results if r.succeeded and r.reply]
        if not successful:
            return {"merged_reply": None, "results": results, "strategy": "ensemble_failed"}
        
        best = max(successful, key=lambda r: len(r.reply))
        
        self.total_distributed_queries += 1
        avg_latency = sum(r.latency_ms for r in successful) / len(successful)
        print(
            f"✅ [DistInfer] 앙상블 완료: {len(successful)}/{len(target_nodes)} 성공, "
            f"최선 노드={best.node_id[:8]}, 평균 지연={avg_latency:.0f}ms"
        )
        
        return {
            "merged_reply": best.reply,
            "best_node": best.node_id,
            "results": results,
            "strategy": "ensemble",
            "avg_latency_ms": avg_latency,
        }
    
    def chain_of_thought_split(self, query: str) -> dict:
        """
        Two-node chain: Node 1 decomposes the problem, Node 2 solves it.
        
        This is the most powerful strategy for complex, multi-step reasoning.
        Node 1 (the "thinker") generates a structured problem breakdown.
        Node 2 (the "solver") receives the breakdown + original query and 
        produces the final answer.
        """
        nodes = self.get_live_nodes()
        if len(nodes) < 2:
            return {"reply": None, "strategy": "chain_not_enough_nodes"}
        
        thinker = nodes[0]
        solver = nodes[1]
        
        # Step 1: Thinker decomposes the problem
        decompose_prompt = f"다음 질문을 단계별로 분해하고 핵심 개념을 나열해주세요 (한국어):\n\n{query}"
        print(f"🧩 [ChainSplit] 분해 노드 ({thinker['node_id'][:8]}) 에 전송 중...")
        think_result = _query_single_node(thinker, decompose_prompt)
        
        if not think_result.succeeded:
            return {"reply": None, "strategy": "chain_thinker_failed", "error": think_result.error}
        
        # Step 2: Solver uses the frame to answer
        solve_prompt = (
            f"[분석 프레임]\n{think_result.reply}\n\n"
            f"[원래 질문]\n{query}\n\n"
            f"위 분석을 바탕으로 최종 답변을 작성해주세요:"
        )
        print(f"🧠 [ChainSplit] 해결 노드 ({solver['node_id'][:8]}) 에 전송 중...")
        solve_result = _query_single_node(solver, solve_prompt)
        
        self.total_distributed_queries += 1
        return {
            "reply": solve_result.reply,
            "thinking_frame": think_result.reply,
            "thinker_node": thinker["node_id"],
            "solver_node": solver["node_id"],
            "strategy": "chain_of_thought_split",
            "total_latency_ms": think_result.latency_ms + solve_result.latency_ms,
        }
    
    def get_stats(self) -> dict:
        return {
            "total_distributed_queries": self.total_distributed_queries,
            "broker": self.broker_url,
        }
