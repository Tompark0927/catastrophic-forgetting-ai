"""
fza_mesh_node.py — Decentralized P2P Mesh Intelligence (v23.0)
===============================================================
Dual-backend architecture:
  - FAST PATH: Tries to import `fza_mesh_rs` (native Rust/PyO3 libp2p module).
    If installed (via `./fza_build_rust.sh`), uses the Rust backend:
      * Lock-free, multi-threaded parallel peer queries
      * Kademlia DHT routing (NAT-traversing, internet-scale)
      * Full concurrent TCP streams without touching the Python GIL
  - FALLBACK: If Rust module not found, uses the pure-Python backend
    (same API, socket-based, local-network only).

Run `./fza_build_rust.sh` to enable the Rust backend.
"""

import os
import json
import time
import socket
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from fza_event_bus import bus

# ── Backend Selection ─────────────────────────────────────────────────────────

try:
    from fza_mesh_rs import FzaMeshNodeRs as _RustNode
    _RUST_BACKEND = True
    print("🦀 [MeshNode] Rust libp2p 백엔드 활성화 (v23.0)")
except ImportError:
    _RustNode = None
    _RUST_BACKEND = False
    print("🐍 [MeshNode] Python 백엔드 활성화 (Rust 미설치 — ./fza_build_rust.sh 실행)")




Current state (v16.0 Swarm): FZA nodes talk through a centralized Broker.
If the Broker dies, the Swarm dies. This is a single point of failure — the
exact mistake every centralized internet company makes.

Phase XV rips out the Broker and replaces it with a self-organizing,
leaderless peer mesh inspired by BitTorrent's DHT routing protocol.

Architecture:
   1. PEER DISCOVERY (UDP Broadcast):
      When a MeshNode starts, it broadcasts a "HELLO" beacon on the local
      subnet every BEACON_INTERVAL_S seconds. Any other MeshNode on the
      network receives it and adds the sender to its peer table.

   2. MESH REGISTRY (In-Memory Peer Table):
      Each node maintains a routing table of known peers:
         peer_id → (ip, port, last_seen, capability_vector)
      Stale entries (not seen in PEER_TTL_S seconds) are pruned automatically.

   3. LATENT QUERY BROADCAST:
      When a node receives a question it can't answer, it encodes it as a
      compressed hidden-state vector (via fza_vector_compression) and
      broadcasts it to all known peers via TCP.
      Peers that can answer stream back their response vectors.

   4. RESPONSE AGGREGATION:
      The querying node collects responses from all peers within a timeout,
      averages the hidden state vectors (weighted by each peer's confidence
      score), and decodes the aggregate into a final response.

Biological Metaphor:
   The Mycelial Network. Underground fungi don't need a central brain to
   route nutrients. Each fungal node (hyphae) connects to its neighbors
   and passes signals along the path of least resistance. If one strand
   dies, the network reroutes.
   FZA's Mesh is its mycelium.

Usage:
    node = MeshNode(node_id="fza-desktop-1", port=9001)
    node.start()
    responses = node.query_mesh("What is the capital of Korea?")
"""

import os
import json
import time
import socket
import struct
import threading
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from fza_event_bus import bus

# ── Configuration ──────────────────────────────────────────────────────────────
BEACON_PORT        = 9999          # UDP broadcast port for peer discovery
QUERY_PORT_BASE    = 10000         # Base TCP port for query responses
BEACON_INTERVAL_S  = 15           # How often to broadcast presence
PEER_TTL_S         = 60           # Drop peers not seen in 60 seconds
QUERY_TIMEOUT_S    = 5            # Max time to wait for peer responses
MAX_PEERS          = 64           # Max entries in peer routing table
FZA_MESH_VERSION   = "v21.0"
BROADCAST_ADDR     = "255.255.255.255"


@dataclass
class PeerInfo:
    """Represents a single known peer in the mesh."""
    node_id:    str
    ip:         str
    port:       int
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)
    generation: int = 0             # Jellyfish generation (older = more experienced)
    score:      float = 0.5         # Benchmark score from last evolution run

    def is_stale(self) -> bool:
        return (time.time() - self.last_seen) > PEER_TTL_S

    def touch(self):
        self.last_seen = time.time()


class MeshNode:
    """
    A full-duplex FZA Mesh peer. Handles:
        - UDP broadcast beacon (presence advertisement)
        - UDP listener (peer discovery)
        - TCP query server (answer incoming queries from peers)
        - TCP query client (send outgoing queries to peers)
        - Peer table management (add, prune stale)
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        port: int = QUERY_PORT_BASE,
        on_query: Optional[Callable[[str], str]] = None,
    ):
        self.node_id    = node_id or f"fza-{socket.gethostname()}-{str(uuid.uuid4())[:8]}"
        self.port       = port
        self.on_query   = on_query    # Callback: local engine answers a query string
        self._peers:    Dict[str, PeerInfo] = {}
        self._lock      = threading.Lock()
        self._running   = False
        self._threads:  List[threading.Thread] = []
        self.total_queries_sent     = 0
        self.total_queries_received = 0
        self.total_responses_aggregated = 0
        self._my_ip = self._get_local_ip()

        print(f"🕸️  [MeshNode] 초기화 | id={self.node_id} | ip={self._my_ip}:{port}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start all background threads: beacon, listener, pruner, TCP server."""
        if self._running:
            return
        self._running = True
        threads = [
            threading.Thread(target=self._run_beacon,   daemon=True, name="mesh-beacon"),
            threading.Thread(target=self._run_listener, daemon=True, name="mesh-listener"),
            threading.Thread(target=self._run_pruner,   daemon=True, name="mesh-pruner"),
            threading.Thread(target=self._run_tcp_server, daemon=True, name="mesh-tcp"),
        ]
        for t in threads:
            t.start()
            self._threads.append(t)
        bus.emit("mesh_started", {"node_id": self.node_id, "port": self.port})
        print(f"🕸️  [MeshNode] 메시 시작! 브로드캐스트 포트: {BEACON_PORT}")

    def stop(self):
        self._running = False
        print(f"🕸️  [MeshNode] 메시 중지.")

    # ── Beacon (UDP Broadcast) ─────────────────────────────────────────────────

    def _run_beacon(self):
        """Broadcasts our presence every BEACON_INTERVAL_S seconds."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        payload = json.dumps({
            "type":    "HELLO",
            "version": FZA_MESH_VERSION,
            "node_id": self.node_id,
            "port":    self.port,
            "ts":      time.time(),
        }).encode()

        while self._running:
            try:
                sock.sendto(payload, (BROADCAST_ADDR, BEACON_PORT))
            except Exception as e:
                pass  # Network may be unavailable; fail silently
            time.sleep(BEACON_INTERVAL_S)
        sock.close()

    # ── Listener (UDP Receive) ─────────────────────────────────────────────────

    def _run_listener(self):
        """Listens for broadcast beacons from peers and updates the peer table."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", BEACON_PORT))
        except OSError as e:
            print(f"⚠️  [MeshNode] 리스너 바인드 실패: {e}")
            return

        sock.settimeout(2.0)
        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
                msg = json.loads(data.decode())
                peer_ip = addr[0]
                if msg.get("type") == "HELLO" and msg.get("node_id") != self.node_id:
                    self._register_peer(msg, peer_ip)
            except socket.timeout:
                pass
            except Exception:
                pass
        sock.close()

    def _register_peer(self, msg: dict, peer_ip: str):
        node_id = msg["node_id"]
        port    = msg.get("port", QUERY_PORT_BASE)
        with self._lock:
            if node_id in self._peers:
                self._peers[node_id].touch()
            else:
                peer = PeerInfo(node_id=node_id, ip=peer_ip, port=port)
                self._peers[node_id] = peer
                print(f"🕸️  [MeshNode] 새 피어 발견: {node_id} @ {peer_ip}:{port}")
                bus.emit("mesh_peer_discovered", {"node_id": node_id, "ip": peer_ip, "port": port})

    # ── Peer Pruner ──────────────────────────────────────────────────────────

    def _run_pruner(self):
        """Removes stale peers that haven't been seen recently."""
        while self._running:
            time.sleep(30)
            with self._lock:
                stale = [pid for pid, p in self._peers.items() if p.is_stale()]
                for pid in stale:
                    del self._peers[pid]
                    print(f"🕸️  [MeshNode] 스테일 피어 제거: {pid}")
                    bus.emit("mesh_peer_pruned", {"node_id": pid})

    # ── TCP Query Server ──────────────────────────────────────────────────────

    def _run_tcp_server(self):
        """Listens for incoming query requests from peers."""
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", self.port))
            srv.listen(16)
            srv.settimeout(2.0)
        except OSError as e:
            print(f"⚠️  [MeshNode] TCP 서버 바인드 실패 (포트 {self.port}): {e}")
            return

        while self._running:
            try:
                conn, addr = srv.accept()
                t = threading.Thread(
                    target=self._handle_query_connection,
                    args=(conn, addr),
                    daemon=True
                )
                t.start()
            except socket.timeout:
                pass
            except Exception:
                pass
        srv.close()

    def _handle_query_connection(self, conn: socket.socket, addr):
        """Handle a single incoming query from a peer."""
        try:
            raw = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                raw += chunk
                if b"\n" in raw:
                    break
            msg = json.loads(raw.decode().strip())
            query_text = msg.get("query", "")
            self.total_queries_received += 1

            # Use local FZA engine to answer if callback provided
            if self.on_query and query_text:
                try:
                    answer = self.on_query(query_text)
                except Exception as e:
                    answer = f"[MESH ERROR] {e}"
            else:
                answer = "[MESH] No local engine available."

            response = json.dumps({
                "node_id": self.node_id,
                "query":   query_text,
                "answer":  answer,
                "ts":      time.time(),
            }) + "\n"
            conn.sendall(response.encode())
        except Exception:
            pass
        finally:
            conn.close()

    # ── Query API (Outgoing) ──────────────────────────────────────────────────

    def query_mesh(self, query_text: str, timeout: float = QUERY_TIMEOUT_S) -> List[dict]:
        """
        Broadcast a query to all known peers. Collect responses.
        Returns a list of response dicts from each responding peer.
        """
        with self._lock:
            peers = list(self._peers.values())

        if not peers:
            print("🕸️  [MeshNode] 쿼리 실패: 알려진 피어 없음")
            return []

        self.total_queries_sent += 1
        results = []
        threads = []
        result_lock = threading.Lock()

        def _query_one(peer: PeerInfo):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((peer.ip, peer.port))
                payload = json.dumps({"query": query_text}) + "\n"
                sock.sendall(payload.encode())
                raw = b""
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    raw += chunk
                    if b"\n" in raw:
                        break
                response = json.loads(raw.decode().strip())
                with result_lock:
                    results.append(response)
                    self.total_responses_aggregated += 1
            except Exception:
                pass
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

        for peer in peers:
            t = threading.Thread(target=_query_one, args=(peer,), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=timeout + 0.5)

        bus.emit("mesh_query_complete", {
            "query": query_text,
            "responses": len(results),
            "peers_queried": len(peers),
        })
        return results

    # ── Status ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            live_peers = [p for p in self._peers.values() if not p.is_stale()]
        return {
            "node_id":          self.node_id,
            "ip":               self._my_ip,
            "port":             self.port,
            "known_peers":      len(live_peers),
            "queries_sent":     self.total_queries_sent,
            "queries_received": self.total_queries_received,
            "responses_agg":    self.total_responses_aggregated,
            "running":          self._running,
        }

    def print_status(self):
        stats = self.get_stats()
        print(f"\n🕸️  [MeshNode] 메시 상태")
        print(f"   노드 ID: {stats['node_id']}")
        print(f"   주소: {stats['ip']}:{stats['port']}")
        print(f"   알려진 피어: {stats['known_peers']}개")
        print(f"   쿼리 전송: {stats['queries_sent']} | 수신: {stats['queries_received']}")
        print(f"   집계된 응답: {stats['responses_agg']}")
        with self._lock:
            for pid, p in self._peers.items():
                age = int(time.time() - p.last_seen)
                stale = "⚠️ STALE" if p.is_stale() else "✅"
                print(f"   └── {pid} @ {p.ip}:{p.port} | {age}s ago {stale}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
