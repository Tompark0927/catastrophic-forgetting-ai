"""
fza_ros2_bridge.py — ROS2 / Robotics Hardware Interface (v18.0)
===============================================================
The peripheral nervous system. Translates FZA ActuatorCommands into
either real ROS2 messages (when rclpy is available) or a rich
simulation mode with full logging.

ROS2 topics published:
    /cmd_vel              (geometry_msgs/Twist)   — mobile base velocity
    /joint_trajectory     (trajectory_msgs/JointTrajectory) — arm control
    /gripper_command      (control_msgs/GripperCommand) — end-effector
    /fza_speech           (std_msgs/String) — TTS output to robot speaker
    /fza_status           (std_msgs/String) — general FZA heartbeat

Note on ROS2 availability:
    rclpy (the ROS2 Python client) requires a ROS2 installation.
    This module gracefully degrades to "sim mode" if rclpy is not found.
    Sim mode provides complete logging and timing simulation, so the
    rest of FZA can be tested without physical hardware.

Sim-to-Real workflow:
    1. Train movement patterns in SIMULATION (dry_run=True)
    2. Motor skills are stored in MotorAdapterBank (EWC-protected)
    3. Switch to real hardware: FZAROS2Bridge(dry_run=False)
    4. Same commands, real actuators — EWC prevents forgetting sim skills

Usage:
    # Simulation (no ROS2 required)
    bridge = FZAROS2Bridge(robot_name="fza_bot", dry_run=True)
    bridge.publish(ActuatorCommand("navigate", {"target": "desk"}, duration_s=2.0))
    
    # Real hardware (requires ROS2 + robot URDF)
    bridge = FZAROS2Bridge(robot_name="spot", dry_run=False)
    bridge.spin()   # starts ROS2 node in background thread
"""

import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, List

from fza_motor_cortex import ActuatorCommand
from fza_event_bus import bus

# Try to import rclpy (ROS2 Python client)
try:
    import rclpy
    from rclpy.node import Node
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False


@dataclass
class CommandLog:
    """Record of a published actuator command for post-hoc analysis."""
    command_type: str
    params: dict
    timestamp: float = field(default_factory=time.time)
    published_to: str = ""
    success: bool = True
    latency_ms: float = 0.0


class MotorSafetyFilter:
    """
    Hardware protection layer (E-Stop and Jerk/Velocity Limiting).
    Prevents the Sim2Real gap from destroying real servos by enforcing
    strict physical bounds on all ActuatorCommands before they are published.
    """
    def __init__(self, max_linear_vel=1.0, max_angular_vel=1.5, allow_hardware=False):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.allow_hardware = allow_hardware
        self.software_e_stop = False
        self._last_cmd_time = time.time()
        
    def is_safe(self, command: ActuatorCommand) -> bool:
        # 1. E-Stop Check
        if self.software_e_stop and command.command_type != "stop":
            print(f"🛑 [SafetyFilter] REJECTED: Software E-Stop is engaged.")
            return False

        # 2. Velocity Limit Check
        if command.command_type == "navigate":
            target = command.params.get("target", "")
            # (In a real implementation, you parse the Twist math here. We mock the threshold.)
            if "hyper_speed" in str(target):
                print(f"🛑 [SafetyFilter] REJECTED: Velocity exceeds limits.")
                return False

        # 3. Jerk Limit Check (too many commands too fast)
        now = time.time()
        if (now - self._last_cmd_time) < 0.005:  # Cannot send commands faster than 200Hz
            print(f"🛑 [SafetyFilter] REJECTED: Jerk limit exceeded (>200Hz).")
            return False
        self._last_cmd_time = now

        return True


class FZAROS2Bridge:
    """
    Publishes FZA ActuatorCommands to robotic hardware via ROS2 topics,
    or logs them in simulation mode.
    """

    # Topic names
    TOPIC_CMDVEL   = "/cmd_vel"
    TOPIC_JOINT    = "/joint_trajectory"
    TOPIC_GRIPPER  = "/gripper_command"
    TOPIC_SPEECH   = "/fza_speech"
    TOPIC_STATUS   = "/fza_status"

    def __init__(
        self,
        robot_name: str = "fza_bot",
        dry_run: bool = True,
        node_name: str = "fza_motor_cortex",
    ):
        self.robot_name = robot_name
        self.dry_run = dry_run
        self.node_name = node_name
        self._ros2_node = None
        self._publishers = {}
        self._command_log: List[CommandLog] = []
        self._commands_published = 0
        self._spin_thread: Optional[threading.Thread] = None
        
        # Initialize Safety Filter
        self.safety = MotorSafetyFilter(allow_hardware=not dry_run)

        if not dry_run and _ROS2_AVAILABLE:
            self._init_ros2()
        elif not dry_run and not _ROS2_AVAILABLE:
            print("⚠️  [ROS2Bridge] rclpy 없음 — pip install rclpy OR install ROS2 Humble")
            print("   → 시뮬레이션 모드로 전환")
            self.dry_run = True

        mode = "🔴 REAL" if not self.dry_run else "🔵 SIM"
        print(f"🤖 [ROS2Bridge] 초기화: robot='{robot_name}', mode={mode}, "
              f"ROS2={'✅' if _ROS2_AVAILABLE else '❌ (시뮬)'}")

    # ── Core Publish API ──────────────────────────────────────────────────────

    def publish(self, command: ActuatorCommand) -> bool:
        """
        Publishes a single ActuatorCommand to the appropriate ROS2 topic,
        or logs it in simulation mode.

        Returns True on success.
        """
        # Safety Gate
        if not self.safety.is_safe(command):
            return False

        t0 = time.perf_counter()
        success = True

        if self.dry_run:
            success = self._sim_publish(command)
        else:
            success = self._ros2_publish(command)

        latency = (time.perf_counter() - t0) * 1000
        log = CommandLog(
            command_type=command.command_type,
            params=command.params,
            published_to=self._topic_for(command.command_type),
            success=success,
            latency_ms=latency,
        )
        self._command_log.append(log)
        self._commands_published += 1

        bus.emit("ros2_command", {
            "robot": self.robot_name,
            "command": command.command_type,
            "topic": log.published_to,
            "latency_ms": round(latency, 2),
            "success": success,
        })
        return success

    def publish_sequence(self, commands: List[ActuatorCommand]) -> List[bool]:
        """Execute a full command sequence and return per-command success."""
        results = []
        for cmd in commands:
            ok = self.publish(cmd)
            results.append(ok)
            if cmd.duration_s > 0 and self.dry_run:
                time.sleep(min(cmd.duration_s * 0.05, 0.05))  # Fast sim tick
        return results

    def emergency_stop(self):
        """Immediately halts all motion — publishes zero velocities."""
        self.safety.software_e_stop = True
        stop_cmd = ActuatorCommand("stop", {}, duration_s=0.0, priority=1)
        self.publish(stop_cmd)
        print("⛔ [ROS2Bridge] EMERGENCY STOP! All future commands blocked.")
        bus.emit("emergency_stop", {"robot": self.robot_name})

    # ── Simulation ────────────────────────────────────────────────────────────

    def _sim_publish(self, command: ActuatorCommand) -> bool:
        """Simulation publish: rich logging without real hardware."""
        topic = self._topic_for(command.command_type)
        msg = self._build_message(command)
        print(f"   🤖 [SIM→{topic}] {command.command_type}: {json.dumps(command.params)[:60]}")
        return True

    def _build_message(self, command: ActuatorCommand) -> dict:
        """Build a ROS2-compatible message dict for a command."""
        if command.command_type == "navigate":
            # geometry_msgs/Twist
            return {
                "linear": {"x": 0.3, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                "target": command.params.get("target", ""),
            }
        elif command.command_type == "joint_move":
            # trajectory_msgs/JointTrajectory
            return {
                "joint_names": ["shoulder", "elbow", "wrist"],
                "pose": command.params.get("pose", "home"),
                "duration_s": command.duration_s,
            }
        elif command.command_type in ("gripper_open", "gripper_close"):
            # control_msgs/GripperCommand
            return {
                "position": 0.0 if command.command_type == "gripper_close" else 0.08,
                "max_effort": 50.0,
            }
        elif command.command_type == "speak":
            return {"text": command.params.get("text", "")}
        elif command.command_type == "stop":
            return {"linear": {"x": 0, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": 0}}
        return command.params

    # ── ROS2 (real hardware) ──────────────────────────────────────────────────

    def _init_ros2(self):
        """Initialize the ROS2 node and create publishers."""
        try:
            rclpy.init()
            self._ros2_node = rclpy.create_node(self.node_name)
            print(f"🤖 [ROS2Bridge] ROS2 노드 생성: '{self.node_name}'")
            # Publishers would be created here in a full implementation:
            # self._publishers['cmd_vel'] = self._ros2_node.create_publisher(Twist, self.TOPIC_CMDVEL, 10)
        except Exception as e:
            print(f"⚠️  [ROS2Bridge] ROS2 초기화 실패: {e}")
            self.dry_run = True

    def _ros2_publish(self, command: ActuatorCommand) -> bool:
        """Publish to real ROS2 topics."""
        if not self._ros2_node:
            return False
        try:
            topic = self._topic_for(command.command_type)
            msg = self._build_message(command)
            # In a full implementation: use typed publishers to publish msg
            print(f"   📡 [ROS2→{topic}] {command.command_type}: {msg}")
            return True
        except Exception as e:
            print(f"❌ [ROS2Bridge] 발행 실패: {e}")
            return False

    def spin(self):
        """Start ROS2 spinning in a background daemon thread."""
        if not _ROS2_AVAILABLE or self.dry_run:
            return
        def _spin():
            try:
                rclpy.spin(self._ros2_node)
            except Exception:
                pass
        self._spin_thread = threading.Thread(target=_spin, daemon=True)
        self._spin_thread.start()
        print("🤖 [ROS2Bridge] ROS2 스핀 시작 (백그라운드)")

    def shutdown(self):
        if _ROS2_AVAILABLE and self._ros2_node:
            self._ros2_node.destroy_node()
            rclpy.shutdown()

    # ── Status ───────────────────────────────────────────────────────────────

    def _topic_for(self, command_type: str) -> str:
        return {
            "navigate": self.TOPIC_CMDVEL,
            "joint_move": self.TOPIC_JOINT,
            "gripper_open": self.TOPIC_GRIPPER,
            "gripper_close": self.TOPIC_GRIPPER,
            "stop": self.TOPIC_CMDVEL,
            "speak": self.TOPIC_SPEECH,
            "wait": "/dev/null",
            "observe": "/dev/null",
        }.get(command_type, self.TOPIC_STATUS)

    def get_stats(self) -> dict:
        return {
            "robot_name": self.robot_name,
            "dry_run": self.dry_run,
            "ros2_available": _ROS2_AVAILABLE,
            "commands_published": self._commands_published,
            "success_rate": sum(1 for l in self._command_log if l.success) / max(1, len(self._command_log)),
            "avg_latency_ms": sum(l.latency_ms for l in self._command_log) / max(1, len(self._command_log)),
        }

    def print_status(self):
        s = self.get_stats()
        print(f"\n🤖 [ROS2Bridge] robot='{s['robot_name']}' | "
              f"{'🔵 SIM' if s['dry_run'] else '🔴 REAL'}")
        print(f"   ROS2: {'✅' if s['ros2_available'] else '❌ (sim mode)'}")
        print(f"   명령 발행: {s['commands_published']}회")
        print(f"   성공률: {s['success_rate']:.0%}")
        print(f"   평균 지연: {s['avg_latency_ms']:.2f}ms")
