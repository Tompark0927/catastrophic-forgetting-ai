"""
fza_swarm_provisioner.py — Asexual Compute Replication (v16.0)
===============================================================
FZA's mitosis engine. When the SelfArchitect detects a sustained
computational overload that KernelForge alone cannot resolve, it
calls the SwarmProvisioner to spawn a child node — an exact clone
of the current FZA instance, running in either a local Docker
container or a remote cloud instance.

Biological Metaphor: Cellular Mitosis.
When a cell is overloaded, it divides. The child cell is a genetic
copy of the parent — same memory (adapters, WorldGraph, procedures),
same specialization (domain lobes), but a fresh compute thread with
no load. The parent delegates overflow work to it immediately.

Four lifecycle states for a child node:
  - SPAWNING: container is being created
  - ALIVE:    registered with cloud broker, serving requests
  - CULLING:  has been idle too long, scheduled for termination
  - DEAD:     container/instance stopped and cleaned up

Backends supported:
  - LOCAL_DOCKER: spins up a Docker container on the local machine
  - MOCK:         dry-run mode (default, no real containers created)
  - (Extensible: AWS EC2, GCP Cloud Run, etc.)

Usage:
    from fza_swarm_provisioner import SwarmProvisioner
    sp = SwarmProvisioner(backend="mock")
    
    node_id = sp.spawn_child("specialized quantum compute node")
    print(sp.list_nodes())
    sp.cull_idle_nodes(max_idle_seconds=300)
"""

import os
import time
import uuid
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from fza_event_bus import bus


# Node lifecycle states
SPAWNING = "spawning"
ALIVE    = "alive"
CULLING  = "culling"
DEAD     = "dead"

# Default thresholds
DEFAULT_MAX_IDLE_SECONDS = 300   # 5 minutes
DEFAULT_MAX_NODES        = 8     # Safety cap on swarm size


@dataclass
class ChildNode:
    """Represents one child FZA node in the swarm."""
    node_id: str
    label: str
    backend: str            # "mock" | "docker" | "ec2"
    status: str = SPAWNING
    host: str = "localhost"
    port: int = 0
    spawned_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    requests_served: int = 0
    container_id: Optional[str] = None   # Docker container ID if applicable

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_active

    @property
    def uptime_minutes(self) -> float:
        return (time.time() - self.spawned_at) / 60

    def to_dict(self) -> dict:
        return asdict(self)


class SwarmProvisioner:
    """
    Manages the lifecycle of child FZA compute nodes.
    Can spawn, monitor, and cull nodes autonomously.
    """

    def __init__(
        self,
        backend: str = "mock",
        base_port: int = 8100,
        max_nodes: int = DEFAULT_MAX_NODES,
        state_file: str = "./swarm_state.json",
    ):
        """
        Args:
            backend:    "mock" (default), "docker", "ec2"
            base_port:  Starting port for child nodes (8100, 8101, ...)
            max_nodes:  Hard cap on total swarm size
            state_file: Path to persist swarm state across restarts
        """
        self.backend = backend
        self.base_port = base_port
        self.max_nodes = max_nodes
        self.state_file = state_file
        self._nodes: Dict[str, ChildNode] = {}
        self._port_counter = base_port
        self._lock = threading.Lock()
        self._load_state()

        # Check backend availability
        self._docker_available = self._check_docker()

        print(f"🦠 [SwarmProvisioner] 초기화 | backend={backend}, "
              f"max_nodes={max_nodes}, "
              f"docker={'✅' if self._docker_available else '❌'}, "
              f"활성 노드: {len([n for n in self._nodes.values() if n.status == ALIVE])}개")

    # ── Public API ────────────────────────────────────────────────────────────

    def spawn_child(self, label: str = "generic") -> Optional[str]:
        """
        Spawns a new child FZA node with the given label/purpose.

        Args:
            label: Human-readable description of what this node will do
                   (e.g. "quantum_physics expert", "vision processing overflow")

        Returns:
            node_id string, or None if max_nodes reached or spawn failed.
        """
        with self._lock:
            alive_count = sum(1 for n in self._nodes.values() if n.status in (SPAWNING, ALIVE))
            if alive_count >= self.max_nodes:
                print(f"⚠️  [SwarmProvisioner] 최대 노드 수 도달 ({self.max_nodes}개) — 분열 거부")
                return None

            node_id = str(uuid.uuid4())[:12]
            port = self._port_counter
            self._port_counter += 1

            node = ChildNode(
                node_id=node_id,
                label=label,
                backend=self.backend,
                host="localhost",
                port=port,
            )
            self._nodes[node_id] = node

        print(f"🔬 [SwarmProvisioner] 분열 시작: '{label}' [{node_id[:8]}] @ port {port}")

        # Dispatch spawn to background thread so it doesn't block
        spawn_thread = threading.Thread(
            target=self._do_spawn,
            args=(node,),
            daemon=True,
        )
        spawn_thread.start()

        return node_id

    def heartbeat(self, node_id: str):
        """A child node calls this to signal it is still alive and processing."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                node.last_active = time.time()
                node.requests_served += 1
                node.status = ALIVE

    def cull_idle_nodes(self, max_idle_seconds: int = DEFAULT_MAX_IDLE_SECONDS) -> List[str]:
        """
        Terminates all child nodes that have been idle for too long.
        Acts as the immune system: prevents zombie processes from draining resources.

        Returns:
            List of node_ids that were culled.
        """
        culled = []
        with self._lock:
            for node in list(self._nodes.values()):
                if node.status == ALIVE and node.idle_seconds > max_idle_seconds:
                    print(f"🗑️  [SwarmProvisioner] 유휴 노드 종료: [{node.node_id[:8]}] "
                          f"'{node.label}' ({node.idle_seconds:.0f}초 유휴)")
                    node.status = CULLING
                    self._do_cull(node)
                    culled.append(node.node_id)

        if culled:
            bus.emit("swarm_culled", {"culled": culled, "count": len(culled)})
        return culled

    def list_nodes(self) -> List[dict]:
        """Returns a snapshot of all node records."""
        with self._lock:
            return [n.to_dict() for n in self._nodes.values()]

    def get_alive_nodes(self) -> List[ChildNode]:
        with self._lock:
            return [n for n in self._nodes.values() if n.status == ALIVE]

    def get_stats(self) -> dict:
        with self._lock:
            statuses = {}
            for n in self._nodes.values():
                statuses[n.status] = statuses.get(n.status, 0) + 1
            return {
                "total_spawned": len(self._nodes),
                "alive": statuses.get(ALIVE, 0),
                "spawning": statuses.get(SPAWNING, 0),
                "dead": statuses.get(DEAD, 0),
                "backend": self.backend,
                "max_nodes": self.max_nodes,
            }

    def print_swarm_status(self):
        nodes = self.list_nodes()
        if not nodes:
            print("🦠 [SwarmProvisioner] 활성 노드 없음 — '세포분열 [이유]'로 새 노드를 생성하세요")
            return
        print(f"\n🦠 [Swarm] {len(nodes)}개 노드:")
        print(f"{'ID':<12} {'상태':<10} {'레이블':<30} {'포트':<6} {'요청수':>6} {'가동(분)':>9}")
        print("-" * 75)
        for n in nodes:
            idle = f"(유휴 {n['idle_seconds']:.0f}초)" if n['status'] == ALIVE and n['idle_seconds'] > 30 else ""
            print(f"{n['node_id'][:10]:<12} {n['status']:<10} {n['label'][:28]:<30} "
                  f"{n['port']:<6} {n['requests_served']:>6} {n['uptime_minutes']:>8.1f}분 {idle}")

    # ── Backend Implementations ───────────────────────────────────────────────

    def _do_spawn(self, node: ChildNode):
        """Executes the actual spawn in a background thread."""
        try:
            if self.backend == "docker" and self._docker_available:
                import subprocess
                # Minimal FZA thin client container
                cmd = [
                    "docker", "run", "-d",
                    "--name", f"fza-child-{node.node_id[:8]}",
                    "-p", f"{node.port}:8000",
                    "-e", f"FZA_NODE_ID={node.node_id}",
                    "-e", f"FZA_BROKER_URL=http://localhost:8001",
                    "python:3.11-slim",
                    "python", "-c",
                    "import http.server; http.server.test(HandlerClass=http.server.BaseHTTPRequestHandler, port=8000)"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    node.container_id = result.stdout.strip()[:12]
                    node.status = ALIVE
                    print(f"🐳 [SwarmProvisioner] Docker 컨테이너 생성: {node.container_id} @ :{node.port}")
                else:
                    raise RuntimeError(result.stderr or "Docker spawn failed")
            else:
                # MOCK backend: simulate spawn delay
                time.sleep(0.5)
                node.status = ALIVE
                print(f"✅ [SwarmProvisioner] [MOCK] 노드 활성화: [{node.node_id[:8]}] '{node.label}' @ port {node.port}")

            self._save_state()
            bus.emit("swarm_spawned", {
                "node_id": node.node_id,
                "label": node.label,
                "port": node.port,
                "backend": self.backend,
            })

        except Exception as e:
            node.status = DEAD
            print(f"❌ [SwarmProvisioner] 분열 실패 [{node.node_id[:8]}]: {e}")

    def _do_cull(self, node: ChildNode):
        """Terminates a node (in background thread for Docker)."""
        try:
            if self.backend == "docker" and node.container_id:
                import subprocess
                subprocess.run(
                    ["docker", "rm", "-f", f"fza-child-{node.node_id[:8]}"],
                    capture_output=True, timeout=10
                )
            node.status = DEAD
            self._save_state()
        except Exception as e:
            print(f"⚠️  [SwarmProvisioner] 종료 실패 [{node.node_id[:8]}]: {e}")

    def _check_docker(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=3)
            return result.returncode == 0
        except Exception:
            return False

    def _save_state(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump({nid: n.to_dict() for nid, n in self._nodes.items()}, f, indent=2)
        except Exception:
            pass

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            for nid, nd in data.items():
                node = ChildNode(**{k: v for k, v in nd.items() if k in ChildNode.__dataclass_fields__})
                # On restart, treat previously-alive nodes as gone (they died with the process)
                if node.status in (SPAWNING, ALIVE):
                    node.status = DEAD
                self._nodes[nid] = node
        except Exception as e:
            print(f"⚠️  [SwarmProvisioner] 상태 로드 실패: {e}")
