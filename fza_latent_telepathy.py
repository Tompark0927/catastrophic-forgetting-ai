"""
fza_latent_telepathy.py — Vector-to-Vector Node Communication (v17.0)
======================================================================
The Unsupervised Protocol Synthesis engine.

Instead of sending English requests (slow, verbose, lossy) over HTTP,
FZA nodes transmit compressed hidden-state tensors directly over UDP.

This is the Latent Telepathy layer: nodes "think at each other" rather
than "speak to each other."

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     TelepathyNode                               │
    │                                                                 │
    │  send_thought(tensor) ──→ compress ──→ UDP socket ──→ peer     │
    │                                                                 │
    │  listen()             ←── UDP recv ←── decompress ←── peer     │
    │                           |                                     │
    │                    on_thought_received(tensor)                  │
    └─────────────────────────────────────────────────────────────────┘

Why UDP (not TCP)?
    - No connection overhead, no ACK round-trips.
    - Dropped packets = slightly degraded hidden states, not catastrophic.
      (LLM inference can tolerate minor approximation errors.)
    - 5–10x lower latency than TCP for small-to-medium payloads.

Packet format (binary):
    [4 bytes: magic "FZAT"]  [4 bytes: payload length]  [N bytes: payload]

Usage:
    from fza_latent_telepathy import TelepathyNode

    # Node A – sender
    nodeA = TelepathyNode(host='0.0.0.0', port=9200)
    nodeA.start_listener()

    # Node B – receiver
    nodeB = TelepathyNode(host='0.0.0.0', port=9201)
    nodeB.start_listener(on_receive=lambda t, addr: print(f'Got tensor {t.shape} from {addr}'))

    # Send a hidden state from A to B
    hidden = torch.randn(8, 4096)
    nodeA.send_thought(hidden, peer_host='localhost', peer_port=9201)

Biological metaphor: The corpus callosum.
The left and right brain hemispheres don't communicate in words — they
transmit compressed neural activation patterns directly. Latent Telepathy
is FZA's corpus callosum: a direct, fast, semantically-dense channel
between compute nodes that bypasses human-readable serialization entirely.
"""

import socket
import struct
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple
import torch

from fza_vector_compression import compress_payload, decompress_payload, compression_ratio
from fza_event_bus import bus

# Packet constants
MAGIC = b"FZAT"             # 4-byte magic header
HEADER_SIZE = 8             # 4 magic + 4 length
MAX_UDP_PAYLOAD = 65000     # UDP practical limit (~65KB)
DEFAULT_QUANT = "fp16"
DEFAULT_PCA = False
DEFAULT_PCA_K = 128


class ThoughtPacket:
    """A single latent vector packet being sent between nodes."""

    def __init__(self, tensor: torch.Tensor, sender_id: str, thought_id: str = None):
        self.tensor = tensor
        self.sender_id = sender_id
        self.thought_id = thought_id or str(uuid.uuid4())[:8]
        self.timestamp = time.time()


class TelepathyNode:
    """
    A single FZA node that can both send and receive latent vector thoughts
    over UDP sockets.

    One node can simultaneously:
    - Listen on its own port for incoming thoughts
    - Broadcast thoughts to known peers
    """

    def __init__(
        self,
        node_id: str = None,
        host: str = "0.0.0.0",
        port: int = 9200,
        quant_mode: str = DEFAULT_QUANT,
        use_pca: bool = DEFAULT_PCA,
        pca_components: int = DEFAULT_PCA_K,
    ):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.host = host
        self.port = port
        self.quant_mode = quant_mode
        self.use_pca = use_pca
        self.pca_components = pca_components

        self._sock: Optional[socket.socket] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._peers: Dict[str, Tuple[str, int]] = {}   # peer_id → (host, port)

        # Stats
        self.thoughts_sent = 0
        self.thoughts_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0

        print(f"📡 [TelepathyNode] 초기화: node_id={self.node_id}, {host}:{port}, "
              f"quant={quant_mode}, pca={use_pca}")

    # ── Peer Management ───────────────────────────────────────────────────────

    def register_peer(self, peer_id: str, peer_host: str, peer_port: int):
        """Register a known peer node so we can broadcast to it by ID."""
        self._peers[peer_id] = (peer_host, peer_port)
        print(f"📡 [Telepathy] 피어 등록: {peer_id} @ {peer_host}:{peer_port}")

    # ── Sending ───────────────────────────────────────────────────────────────

    def send_thought(
        self,
        tensor: torch.Tensor,
        peer_host: str = None,
        peer_port: int = None,
        peer_id: str = None,
    ) -> bool:
        """
        Compresses and transmits a hidden-state tensor to a peer node via UDP.

        Args:
            tensor:    The hidden state to transmit [any shape]
            peer_host: Destination host (if not using peer_id registry)
            peer_port: Destination port
            peer_id:   Registered peer ID (alternative to host/port)

        Returns:
            True if sent successfully, False on error.
        """
        # Resolve destination
        if peer_id and peer_id in self._peers:
            peer_host, peer_port = self._peers[peer_id]
        if not peer_host or not peer_port:
            print("⚠️  [Telepathy] 목적지 없음 — peer_host/peer_port 또는 peer_id 필요")
            return False

        try:
            # Compress the tensor
            payload = compress_payload(
                tensor,
                quant_mode=self.quant_mode,
                use_pca=self.use_pca,
                pca_components=self.pca_components,
            )

            # Build packet: [MAGIC][4-byte length][payload]
            length_bytes = struct.pack(">I", len(payload))
            packet = MAGIC + length_bytes + payload

            if len(packet) > MAX_UDP_PAYLOAD:
                # Fragment: for simplicity, truncate & warn in this version
                print(f"⚠️  [Telepathy] 패킷 너무 큼 ({len(packet)}B > {MAX_UDP_PAYLOAD}B) — 압축 강도 증가 필요")
                return False

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(packet, (peer_host, peer_port))
            sock.close()

            ratio = compression_ratio(tensor, payload)
            self.thoughts_sent += 1
            self.bytes_sent += len(packet)

            bus.emit("telepathy_sent", {
                "sender": self.node_id,
                "peer": f"{peer_host}:{peer_port}",
                "tensor_shape": list(tensor.shape),
                "bytes": len(packet),
                "compression_ratio": round(ratio, 2),
            })

            print(f"🧠→ [Telepathy] 전송: {tensor.shape} → {peer_host}:{peer_port} "
                  f"| {len(packet)}B (압축률 {ratio:.1f}x)")
            return True

        except Exception as e:
            print(f"❌ [Telepathy] 전송 실패: {e}")
            return False

    def broadcast_thought(self, tensor: torch.Tensor) -> int:
        """Broadcasts a thought to ALL registered peers. Returns count sent."""
        sent = 0
        for pid, (h, p) in self._peers.items():
            if self.send_thought(tensor, peer_host=h, peer_port=p):
                sent += 1
        return sent

    # ── Receiving ─────────────────────────────────────────────────────────────

    def start_listener(
        self,
        on_receive: Optional[Callable[[torch.Tensor, str], None]] = None,
        timeout: float = 1.0,
    ):
        """
        Starts a background UDP listener thread.

        Args:
            on_receive: Callback called when a tensor arrives: on_receive(tensor, sender_addr)
            timeout:    Socket timeout in seconds (controls responsiveness to stop signal)
        """
        if self._running:
            print("⚠️  [Telepathy] 리스너가 이미 실행 중입니다")
            return

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(timeout)
        self._running = True

        def _listen_loop():
            print(f"📡 [Telepathy] 리스너 시작: {self.host}:{self.port}")
            while self._running:
                try:
                    data, addr = self._sock.recvfrom(MAX_UDP_PAYLOAD + HEADER_SIZE)
                    self._handle_packet(data, addr, on_receive)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        print(f"⚠️  [Telepathy] 수신 오류: {e}")

        self._listener_thread = threading.Thread(target=_listen_loop, daemon=True)
        self._listener_thread.start()

    def stop_listener(self):
        """Gracefully stops the UDP listener thread."""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        print(f"📡 [Telepathy] 리스너 종료: {self.host}:{self.port}")

    def _handle_packet(
        self,
        data: bytes,
        addr: tuple,
        on_receive: Optional[Callable],
    ):
        """Validates and decodes a received UDP packet."""
        if len(data) < HEADER_SIZE:
            return

        if data[:4] != MAGIC:
            return  # Not an FZA telepathy packet

        payload_len = struct.unpack(">I", data[4:8])[0]
        payload = data[8:8 + payload_len]

        try:
            tensor = decompress_payload(payload)
            self.thoughts_received += 1
            self.bytes_received += len(data)

            sender_addr = f"{addr[0]}:{addr[1]}"
            bus.emit("telepathy_received", {
                "receiver": self.node_id,
                "sender": sender_addr,
                "tensor_shape": list(tensor.shape),
                "bytes": len(data),
            })

            print(f"🧠← [Telepathy] 수신: {tensor.shape} ← {sender_addr}")

            if on_receive:
                on_receive(tensor, sender_addr)

        except Exception as e:
            print(f"⚠️  [Telepathy] 패킷 복원 실패: {e}")

    # ── Status ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "running": self._running,
            "peers": len(self._peers),
            "thoughts_sent": self.thoughts_sent,
            "thoughts_received": self.thoughts_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }

    def print_status(self):
        s = self.get_stats()
        print(f"\n📡 [TelepathyNode] {s['node_id']} @ {s['host']}:{s['port']}")
        print(f"   상태: {'🟢 LIVE' if s['running'] else '🔴 OFF'}")
        print(f"   피어: {s['peers']}개")
        print(f"   전송: {s['thoughts_sent']}회 ({s['bytes_sent']:,}B)")
        print(f"   수신: {s['thoughts_received']}회 ({s['bytes_received']:,}B)")
