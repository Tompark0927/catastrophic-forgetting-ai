"""
fza_node_registry.py — Live Node Catalog for the Hive-Mind (v11.0)
====================================================================
Tracks all live FZA nodes in the Mycorrhizal Network: their specializations
(which adapter topics they're expert in), current load, and latency.

Each FZA node registers itself when it boots and sends a heartbeat every 30s.
The broker uses this catalog to answer the dMoE router's question:
  "Which node in the network is the best expert for this query?"

A node's "expertise" is represented as a list of topic embeddings derived from
the node's adapter bank. When a query comes in, the router computes cosine
similarity against all stored topic embeddings and picks the top-K expert nodes.
"""

import time
import uuid
from typing import Dict, List, Optional


class NodeRecord:
    """Represents a single registered FZA node."""
    
    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        adapter_count: int = 0,
        adapter_topics: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.adapter_count = adapter_count
        self.adapter_topics = adapter_topics or []   # list of plain-text topic strings
        self.device = device                          # "cpu", "mps", "cuda"
        self.registered_at = time.time()
        self.last_heartbeat = time.time()
        self.latency_ms: float = 0.0                  # Round-trip latency (measured by broker)
        self.load: float = 0.0                        # 0.0 = idle, 1.0 = saturated
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_alive(self) -> bool:
        """A node is considered alive if it sent a heartbeat within the last 90 seconds."""
        return (time.time() - self.last_heartbeat) < 90
    
    def update_heartbeat(self, adapter_count: int = None, load: float = None):
        self.last_heartbeat = time.time()
        if adapter_count is not None:
            self.adapter_count = adapter_count
        if load is not None:
            self.load = load
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "adapter_count": self.adapter_count,
            "adapter_topics": self.adapter_topics,
            "device": self.device,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "is_alive": self.is_alive,
            "latency_ms": self.latency_ms,
            "load": self.load,
            "base_url": self.base_url,
        }


class NodeRegistry:
    """
    In-memory registry of all live FZA nodes in the Hive-Mind.
    
    Used by:
    - The broker to route distributed inference requests.
    - The dMoE router to find expert nodes.
    - The UI to display a live network topology map.
    """
    
    def __init__(self):
        self._nodes: Dict[str, NodeRecord] = {}
    
    def register(
        self,
        host: str,
        port: int,
        adapter_count: int = 0,
        adapter_topics: List[str] = None,
        device: str = "cpu",
        node_id: str = None,
    ) -> NodeRecord:
        """Registers a new node or refreshes an existing one."""
        nid = node_id or str(uuid.uuid4())
        
        if nid in self._nodes:
            node = self._nodes[nid]
            node.update_heartbeat(adapter_count=adapter_count)
            node.adapter_topics = adapter_topics or node.adapter_topics
        else:
            node = NodeRecord(
                node_id=nid,
                host=host,
                port=port,
                adapter_count=adapter_count,
                adapter_topics=adapter_topics or [],
                device=device,
            )
            self._nodes[nid] = node
            print(f"🔗 [NodeRegistry] 새 노드 등록: {nid[:8]} ({host}:{port}, device={device}, {adapter_count}개 어댑터)")
        
        return node
    
    def heartbeat(self, node_id: str, adapter_count: int = None, load: float = None) -> bool:
        """Receives a heartbeat from a node. Returns False if node is not registered."""
        if node_id not in self._nodes:
            return False
        self._nodes[node_id].update_heartbeat(adapter_count=adapter_count, load=load)
        return True
    
    def unregister(self, node_id: str):
        """Removes a node from the registry."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            print(f"🔌 [NodeRegistry] 노드 해제: {node_id[:8]}")
    
    def get_alive(self) -> List[NodeRecord]:
        """Returns all currently alive nodes."""
        return [n for n in self._nodes.values() if n.is_alive]
    
    def get_all(self) -> List[NodeRecord]:
        return list(self._nodes.values())
    
    def get(self, node_id: str) -> Optional[NodeRecord]:
        return self._nodes.get(node_id)
    
    def find_experts(self, query_topics: List[str], top_k: int = 2) -> List[NodeRecord]:
        """
        Finds the top-K nodes most relevant to the given query topics.
        
        Uses a simple set-overlap score: how many of the node's topics
        match the query topics. In production this would use vector cosine
        similarity via sentence-transformers.
        
        Args:
            query_topics: List of topic strings extracted from the user query
            top_k:        Number of expert nodes to return
        
        Returns:
            List of NodeRecord sorted by relevance (most relevant first)
        """
        alive = self.get_alive()
        if not alive:
            return []
        
        query_set = set(t.lower() for t in query_topics)
        
        def _score(node: NodeRecord) -> float:
            # Overlap score
            node_set = set(t.lower() for t in node.adapter_topics)
            overlap = len(query_set & node_set)
            # Penalize by current load (prefer idle nodes)
            return overlap * (1.0 - node.load * 0.5)
        
        scored = sorted(alive, key=_score, reverse=True)
        return scored[:top_k]
    
    def summary(self) -> dict:
        alive = self.get_alive()
        total = len(self._nodes)
        return {
            "total_registered": total,
            "alive": len(alive),
            "dead": total - len(alive),
            "nodes": [n.to_dict() for n in alive],
        }


# Module-level singleton for use within the broker process
registry = NodeRegistry()
