"""
fza_procedural_memory.py — EWC-Protected Procedural Workflow Memory (v15.0)
=============================================================================
When FZA successfully completes an OS task (e.g., "open Safari and navigate to YouTube"),
it doesn't just forget how it did it. This module records the workflow as a
Procedural Adapter — a compact, persistent record of the exact action sequence —
protected by EWC so the agent never un-learns how to use an application.

Biological Metaphor: Procedural Memory in the Basal Ganglia.
When a pianist learns a scale, the motor pattern is encoded in the basal ganglia
and cerebellum, not the hippocampus. It becomes automatic, unconscious, fast.
This module is FZA's basal ganglia: motor patterns are stored separately from
episodic facts and are extremely resistant to forgetting.

Key design decisions:
  - Workflows are stored as JSON sequences of ActionSteps.
  - Each workflow has a "mastery score" (0 to 100) that increases each time
    the workflow is executed successfully. Higher mastery → higher priority retrieval.
  - EWC protection: workflows with high mastery are "locked" and cannot be
    overwritten by similar but conflicting workflows.

Usage:
    from fza_procedural_memory import ProceduralMemory
    pm = ProceduralMemory()
    
    # Record a successful workflow
    pm.record_workflow("open Safari", steps=[...], success=True)
    
    # Retrieve the best known plan for a new goal
    plan = pm.retrieve("open Chrome")
    if plan:
        agent.execute_task("open Chrome", plan=plan)
    
    # Print all known procedures
    pm.print_summary()
"""

import os
import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict


PROCEDURES_DIR = "./procedures"


@dataclass
class ProceduralAdapter:
    """A single recorded workflow."""
    adapter_id: str
    goal_description: str
    keywords: List[str]         # Derived from goal_description for fuzzy search
    steps: List[dict]           # Serialized ActionStep dicts
    mastery_score: float = 0.0  # 0–100, increases with successful executions
    execution_count: int = 0
    success_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    ewc_locked: bool = False    # True when mastery >= 80 (EWC protection kicks in)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.execution_count)

    def matches(self, query: str) -> float:
        """Fuzzy match: returns a similarity score [0, 1] against a query string."""
        q_lower = query.lower()
        q_words = set(q_lower.split())
        keyword_overlap = len(q_words & set(self.keywords))
        return min(1.0, keyword_overlap / max(1, len(q_words))) * self.success_rate

    def to_dict(self) -> dict:
        return asdict(self)


class ProceduralMemory:
    """
    Persistent memory bank for OS workflows (Procedural Adapters).
    Workflows are stored under `./procedures/` and loaded at startup.
    """

    MASTERY_EWC_LOCK = 80.0      # Lock workflows above this mastery
    MASTERY_INCREMENT = 10.0     # How much mastery increases per success
    MASTERY_DECAY = 2.0          # How much mastery decreases per failure

    def __init__(self, procedures_dir: str = PROCEDURES_DIR):
        self.procedures_dir = procedures_dir
        os.makedirs(procedures_dir, exist_ok=True)
        self._adapters: Dict[str, ProceduralAdapter] = {}
        self._load_all()
        print(f"🧠 [ProceduralMemory] {len(self._adapters)}개 프로시저 로드 완료")

    # ── Public API ────────────────────────────────────────────────────────────

    def record_workflow(self, goal: str, steps: list, success: bool) -> ProceduralAdapter:
        """
        Records (or updates) a workflow for a goal.

        If a similar workflow already exists, updates its mastery score.
        Otherwise, creates a new ProceduralAdapter.
        """
        # Check if we have a similar workflow
        existing = self._find_best_match(goal, threshold=0.5)

        if existing:
            # Update the existing adapter
            existing.execution_count += 1
            existing.last_used = time.time()
            if success:
                existing.success_count += 1
                existing.mastery_score = min(100.0, existing.mastery_score + self.MASTERY_INCREMENT)
                # EWC lock at high mastery
                if existing.mastery_score >= self.MASTERY_EWC_LOCK:
                    existing.ewc_locked = True
                    print(f"🔒 [ProceduralMemory] EWC 잠금: '{goal}' (숙련도 {existing.mastery_score:.0f})")
                else:
                    existing.mastery_score = max(0.0, existing.mastery_score - self.MASTERY_DECAY)
            self._save(existing)
            adapter = existing
        else:
            # Create new adapter
            keywords = [w.lower() for w in goal.split() if len(w) > 2]
            steps_data = []
            for s in steps:
                if hasattr(s, '__dict__'):
                    steps_data.append(asdict(s) if hasattr(s, '__dataclass_fields__') else s.__dict__)
                elif isinstance(s, dict):
                    steps_data.append(s)

            adapter = ProceduralAdapter(
                adapter_id=str(uuid.uuid4())[:12],
                goal_description=goal,
                keywords=keywords,
                steps=steps_data,
                mastery_score=self.MASTERY_INCREMENT if success else 0.0,
                execution_count=1,
                success_count=1 if success else 0,
            )
            self._adapters[adapter.adapter_id] = adapter
            self._save(adapter)
            print(f"📝 [ProceduralMemory] 새 프로시저: '{goal}' ({len(steps_data)}단계)")

        return adapter

    def retrieve(self, goal: str, top_k: int = 1) -> Optional[list]:
        """
        Retrieves the best-matching workflow steps for a given goal.
        Returns the raw steps list, or None if no good match is found.
        """
        best = self._find_best_match(goal, threshold=0.3)
        if not best:
            return None
        # Deserialize steps
        from fza_os_agent import ActionStep
        try:
            return [ActionStep(**s) for s in best.steps]
        except Exception:
            return best.steps

    def get_all_procedures(self) -> List[ProceduralAdapter]:
        return sorted(self._adapters.values(), key=lambda a: a.mastery_score, reverse=True)

    def print_summary(self):
        procedures = self.get_all_procedures()
        if not procedures:
            print("🧠 [ProceduralMemory] 저장된 프로시저 없음")
            return
        print(f"\n🧠 [ProceduralMemory] {len(procedures)}개 프로시저:")
        print(f"{'목표':<35} {'숙련도':>8} {'성공률':>8} {'실행':>6} {'잠금':>6}")
        print("-" * 70)
        for a in procedures:
            lock = "🔒" if a.ewc_locked else "  "
            print(f"{a.goal_description[:34]:<35} {a.mastery_score:>7.0f}% {a.success_rate:>7.0%} {a.execution_count:>6} {lock:>6}")

    def get_stats(self) -> dict:
        locked = sum(1 for a in self._adapters.values() if a.ewc_locked)
        return {
            "total_procedures": len(self._adapters),
            "ewc_locked": locked,
            "avg_mastery": sum(a.mastery_score for a in self._adapters.values()) / max(1, len(self._adapters)),
        }

    # ── Private ────────────────────────────────────────────────────────────────

    def _find_best_match(self, goal: str, threshold: float = 0.3) -> Optional[ProceduralAdapter]:
        best_score = 0.0
        best_adapter = None
        for adapter in self._adapters.values():
            score = adapter.matches(goal)
            if score > best_score:
                best_score = score
                best_adapter = adapter
        return best_adapter if best_score >= threshold else None

    def _save(self, adapter: ProceduralAdapter):
        path = os.path.join(self.procedures_dir, f"{adapter.adapter_id}.json")
        with open(path, "w") as f:
            json.dump(adapter.to_dict(), f, ensure_ascii=False, indent=2)

    def _load_all(self):
        for fname in os.listdir(self.procedures_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self.procedures_dir, fname)) as f:
                        data = json.load(f)
                    adapter = ProceduralAdapter(**data)
                    self._adapters[adapter.adapter_id] = adapter
                except Exception as e:
                    print(f"⚠️  [ProceduralMemory] 로드 실패 ({fname}): {e}")
