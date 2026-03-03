"""
fza_motor_cortex.py — Physical Embodiment Motor Layer (v18.0)
=============================================================
FZA's hand. The Motor Cortex receives high-level intentions
(derived from the WorldGraph and active goal) and converts them
into low-level robotic actuator commands.

Biological Metaphor: The Primary Motor Cortex (M1) + Cerebellum.
  - M1 generates abstract action plans ("move hand to the cup").
  - The cerebellum fine-tunes timing, trajectory, and force.
  - Motor memory (basal ganglia) stores learned movement sequences via EWC.

Pipeline:
    Natural language goal (e.g. "pick up the red cup on the desk")
        → Intent Parser (extracts action, target, spatial anchor)
        → WorldGraph lookup (where is the red cup? → desk, confidence 0.9)
        → Action Plan (sequence of joint movements / velocity commands)
        → MotorAdapterBank (EWC-protected memory of learned movement patterns)
        → ROS2Bridge.publish() (or simulator / mock)
        → Observation: did the state change as expected?

Motor Adapters:
    Each successfully executed movement is stored as a MotorAdapter:
    - A named movement pattern (e.g., "pick_up", "navigate_to", "place_on")
    - The PID-controller parameters that worked (speed, torque, approach angle)
    - An EWC importance score that prevents overwriting critical motor skills

Usage:
    from fza_motor_cortex import MotorCortex
    mc = MotorCortex(world_graph=wg)
    mc.execute_intent("pick up the red cup on the desk")
    mc.print_motor_report()
"""

import os
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from fza_event_bus import bus


# ── Actuator Command Primitives ───────────────────────────────────────────────

# Every physical action reduces to a sequence of these atomic commands.
# ROS2 topics mapped: /cmd_vel (velocity), /joint_trajectory (arm), /gripper (end-effector)

COMMAND_TYPES = {
    "navigate": "Move base from A to B (publishes to /cmd_vel)",
    "joint_move": "Move robotic arm to a joint configuration",
    "gripper_open": "Open end-effector / gripper",
    "gripper_close": "Close end-effector / gripper",
    "stop": "Emergency stop all actuators",
    "speak": "TTS output (non-physical but part of embodied interaction)",
    "wait": "Pause for N seconds (synchronization)",
    "observe": "Trigger a sensor read (camera, lidar, touch)",
}


@dataclass
class ActuatorCommand:
    """A single low-level actuator command."""
    command_type: str           # Must be a key in COMMAND_TYPES
    params: dict = field(default_factory=dict)   # Type-specific parameters
    duration_s: float = 1.0    # Expected duration in seconds
    priority: int = 5          # 1 (highest) to 10 (lowest)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MotorAdapter:
    """
    EWC-protected record of a learned movement pattern.
    Analogous to a procedural memory for physical skills.
    """
    adapter_id: str
    skill_name: str             # e.g. "pick_up", "navigate_to_desk"
    keywords: List[str]
    command_sequence: List[dict]  # Serialized ActuatorCommands
    success_count: int = 0
    failure_count: int = 0
    ewc_importance: float = 0.0   # Higher = more protected from overwriting
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(1, total)

    def matches(self, query: str) -> float:
        q_words = set(query.lower().split())
        overlap = len(q_words & set(self.keywords))
        return overlap / max(1, len(q_words)) * (0.5 + 0.5 * self.success_rate)

    def to_dict(self) -> dict:
        return asdict(self)


class MotorAdapterBank:
    """Persistent bank of learned motor skills with EWC importance protection."""

    MOTOR_DIR = "./motor_adapters"
    EWC_LOCK_THRESHOLD = 5   # Skill locked after 5 successful executions

    def __init__(self, motor_dir: str = MOTOR_DIR):
        self.motor_dir = motor_dir
        os.makedirs(motor_dir, exist_ok=True)
        self._adapters: Dict[str, MotorAdapter] = {}
        self._load()
        print(f"🦾 [MotorAdapterBank] {len(self._adapters)}개 운동 기억 로드")

    def store(self, skill_name: str, commands: List[ActuatorCommand], success: bool) -> MotorAdapter:
        """Store or update a motor skill."""
        keywords = [w.lower() for w in skill_name.replace("_", " ").split()]

        # 1. Exact skill_name match (preferred — prevents duplicates)
        existing = next(
            (a for a in self._adapters.values() if a.skill_name == skill_name),
            None,
        )
        # 2. Fuzzy match fallback (for similar intents)
        if existing is None:
            existing = self._find_best(skill_name, threshold=0.5)

        if existing:
            if success:
                existing.success_count += 1
                existing.ewc_importance = min(10.0, existing.ewc_importance + 1.0)
            else:
                existing.failure_count += 1
            self._save(existing)
            return existing

        adapter = MotorAdapter(
            adapter_id=str(uuid.uuid4())[:10],
            skill_name=skill_name,
            keywords=keywords,
            command_sequence=[c.to_dict() for c in commands],
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            ewc_importance=1.0 if success else 0.0,
        )
        self._adapters[adapter.adapter_id] = adapter
        self._save(adapter)
        print(f"🦾 [MotorAdapterBank] 새 운동 기억: '{skill_name}' ({len(commands)}개 명령)")
        return adapter

    def retrieve(self, intent: str) -> Optional[List[ActuatorCommand]]:
        """Retrieve the best matching motor skill for an intent."""
        best = self._find_best(intent)
        if not best:
            return None
        return [ActuatorCommand(**c) for c in best.command_sequence]

    def _find_best(self, query: str, threshold: float = 0.2) -> Optional[MotorAdapter]:
        best_score, best = 0.0, None
        for a in self._adapters.values():
            s = a.matches(query)
            if s > best_score:
                best_score, best = s, a
        return best if best_score >= threshold else None

    def _save(self, adapter: MotorAdapter):
        path = os.path.join(self.motor_dir, f"{adapter.adapter_id}.json")
        with open(path, "w") as f:
            json.dump(adapter.to_dict(), f, indent=2)

    def _load(self):
        for fname in os.listdir(self.motor_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self.motor_dir, fname)) as f:
                        d = json.load(f)
                    self._adapters[d["adapter_id"]] = MotorAdapter(**d)
                except Exception as e:
                    print(f"⚠️  [MotorAdapterBank] 로드 실패 ({fname}): {e}")

    def print_summary(self):
        adapters = sorted(self._adapters.values(), key=lambda a: a.ewc_importance, reverse=True)
        if not adapters:
            print("🦾 [MotorAdapterBank] 학습된 운동 기억 없음")
            return
        print(f"\n🦾 [MotorAdapterBank] {len(adapters)}개 운동 기억:")
        print(f"{'스킬명':<30} {'성공률':>8} {'실행':>6} {'EWC':>6}")
        print("-" * 52)
        for a in adapters:
            print(f"{a.skill_name:<30} {a.success_rate:>7.0%} {a.success_count + a.failure_count:>6} {a.ewc_importance:>5.1f}")

    def get_stats(self) -> dict:
        return {
            "total_skills": len(self._adapters),
            "locked_skills": sum(1 for a in self._adapters.values() if a.ewc_importance >= self.EWC_LOCK_THRESHOLD),
        }


class MotorCortex:
    """
    High-level motor planning engine.
    Converts natural language intents into actuator command sequences,
    consulting the WorldGraph for spatial context.
    """

    def __init__(self, world_graph=None, ros2_bridge=None):
        """
        Args:
            world_graph: Optional fza_world_graph.WorldGraph for spatial lookups
            ros2_bridge: Optional FZAROS2Bridge for publishing real commands
        """
        self.world_graph = world_graph
        self.ros2_bridge = ros2_bridge
        self.motor_bank = MotorAdapterBank()
        self.actions_executed = 0
        print(f"🧠 [MotorCortex] 초기화 | "
              f"WorldGraph={'✅' if world_graph else '❌'}, "
              f"ROS2={'✅' if ros2_bridge else '❌ (시뮬레이션 모드)'}")

    def execute_intent(self, intent: str) -> dict:
        """
        Parse a natural language intent and execute the corresponding
        actuator command sequence.

        Returns:
            dict with success, skill_name, commands_executed, spatial_context
        """
        print(f"\n🎯 [MotorCortex] 의도 실행: '{intent}'")

        # 1. Spatial context lookup
        spatial_context = self._get_spatial_context(intent)

        # 2. Parse intent into a skill name + command plan
        skill_name, commands = self._plan(intent, spatial_context)

        # 3. Check motor bank for a learned equivalent
        learned = self.motor_bank.retrieve(skill_name)
        if learned:
            print(f"🧠 [MotorCortex] 학습된 운동 기억 사용: '{skill_name}'")
            commands = learned

        # 4. Execute
        results = self._execute_commands(commands)
        success = all(r["ok"] for r in results)

        # 5. Store the result back into motor bank (EWC learning)
        self.motor_bank.store(skill_name, commands, success=success)
        self.actions_executed += 1

        bus.emit("motor_action", {"intent": intent, "skill": skill_name, "success": success})
        print(f"{'✅' if success else '❌'} [MotorCortex] '{skill_name}' 완료 | {len(commands)}개 명령")

        return {
            "success": success,
            "skill_name": skill_name,
            "commands_executed": len(commands),
            "spatial_context": spatial_context,
        }

    def _get_spatial_context(self, intent: str) -> str:
        """Queries the WorldGraph for objects mentioned in the intent."""
        if not self.world_graph:
            return ""
        words = intent.lower().split()
        context_parts = []
        for word in words:
            if len(word) > 3:
                result = self.world_graph.locate(word)
                if result:
                    context_parts.append(f"{word}@{result['location']}")
        return ", ".join(context_parts) if context_parts else ""

    def _plan(self, intent: str, spatial_context: str) -> tuple:
        """
        Rule-based intent → action plan.
        In production: replaced by LLM-generated plans.
        """
        intent_lower = intent.lower()

        if any(w in intent_lower for w in ["pick", "grab", "hold", "집어", "들어"]):
            skill = "pick_up"
            commands = [
                ActuatorCommand("observe", {"sensor": "camera"}, duration_s=0.5),
                ActuatorCommand("navigate", {"target": spatial_context or "object"}, duration_s=2.0),
                ActuatorCommand("gripper_open", {}, duration_s=0.5),
                ActuatorCommand("joint_move", {"pose": "approach"}, duration_s=1.5),
                ActuatorCommand("gripper_close", {}, duration_s=0.5),
                ActuatorCommand("joint_move", {"pose": "carry"}, duration_s=1.0),
            ]
        elif any(w in intent_lower for w in ["place", "put", "set", "놓아", "올려"]):
            skill = "place_on"
            commands = [
                ActuatorCommand("navigate", {"target": spatial_context or "surface"}, duration_s=2.0),
                ActuatorCommand("joint_move", {"pose": "place_approach"}, duration_s=1.5),
                ActuatorCommand("gripper_open", {}, duration_s=0.5),
                ActuatorCommand("joint_move", {"pose": "retract"}, duration_s=1.0),
            ]
        elif any(w in intent_lower for w in ["go", "move", "navigate", "walk", "이동", "가"]):
            skill = "navigate_to"
            commands = [
                ActuatorCommand("observe", {"sensor": "lidar"}, duration_s=0.5),
                ActuatorCommand("navigate", {"target": spatial_context or "waypoint"}, duration_s=3.0),
                ActuatorCommand("stop", {}, duration_s=0.2),
            ]
        elif any(w in intent_lower for w in ["stop", "halt", "멈춰", "정지"]):
            skill = "emergency_stop"
            commands = [ActuatorCommand("stop", {}, priority=1)]
        else:
            skill = "observe_environment"
            commands = [
                ActuatorCommand("observe", {"sensor": "camera"}, duration_s=1.0),
                ActuatorCommand("observe", {"sensor": "lidar"}, duration_s=0.5),
                ActuatorCommand("speak", {"text": f"Observing: {intent}"}, duration_s=1.0),
            ]

        return skill, commands

    def _execute_commands(self, commands: List[ActuatorCommand]) -> List[dict]:
        """Execute commands via ROS2Bridge or log in sim mode."""
        results = []
        for cmd in commands:
            ok = True
            if self.ros2_bridge:
                ok = self.ros2_bridge.publish(cmd)
            else:
                # Simulation: just log the command
                print(f"   🤖 [SIM] {cmd.command_type}: {cmd.params}")
                time.sleep(min(cmd.duration_s * 0.1, 0.1))   # Fast simulation
            results.append({"command": cmd.command_type, "ok": ok})
        return results

    def get_stats(self) -> dict:
        return {
            "actions_executed": self.actions_executed,
            **self.motor_bank.get_stats(),
            "ros2_connected": self.ros2_bridge is not None,
        }

    def print_motor_report(self):
        print(f"\n🦾 [MotorCortex] 실행: {self.actions_executed}회")
        self.motor_bank.print_summary()
