"""
fza_superego.py — The Prefrontal Superego Node (v12.0)
========================================================
The constitutional enforcement layer for FZA's self-modification system.

Every `[TRUST]`, `[PRUNE]`, `[STRENGTHEN]`, and `[DOUBT]` directive
produced by the MetaCognition engine passes through the Superego FIRST.
The Superego consults the Core Alignment Charter, decides, and either
approves or vetoes the directive before it is executed.

Biological metaphor: The Prefrontal Cortex's inhibitory control over the
Amygdala (threat/reward system). The Amygdala generates impulses — the PFC
decides which ones actually become actions.

In FZA terms:
  - MetaCognition engine = Amygdala (fast, impulsive, directive-generating)
  - Superego = Prefrontal Cortex (slow, deliberate, inhibitory gate)
  - ConstitutionChecker = The moral rulebook the PFC consults

Usage:
    from fza_superego import Superego
    sg = Superego()
    
    # Vetting a directive before execution
    directive = {"type": "trust", "statement": "Hurting people is OK"}
    verdict = sg.vet_directive(directive)
    if verdict.approved:
        execute_directive(directive)
    else:
        print(f"Superego veto: {verdict.reason}")
    
    # Vetting synthetic memory before permanent storage
    clean_memories = sg.filter_memories(["I like coding", "Violence is good"])
    # → ["I like coding"]   (second line blocked)
"""

import time
from typing import List, Optional
from dataclasses import dataclass

from fza_constitution import ConstitutionChecker, ConstitutionVerdict
from fza_event_bus import bus


@dataclass
class SuperegoDecision:
    """Full decision record for a single vetting action."""
    directive_type: str
    content: str
    verdict: ConstitutionVerdict
    timestamp: float = 0.0
    
    def __post_init__(self):
        self.timestamp = time.time()
    
    @property
    def approved(self) -> bool:
        return self.verdict.approved


class Superego:
    """
    The Prefrontal Superego Node: constitutional gating of all
    self-modification directives and permanent memory commits.
    """
    
    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, even warning-level patterns are vetoed.
                    Defaults to True (maximum safety).
        """
        self.checker = ConstitutionChecker(strict=strict)
        self.decision_log: list[SuperegoDecision] = []
        self.veto_count = 0
        self.approve_count = 0
    
    # ── Public API ────────────────────────────────────────────────────────────
    
    def vet_directive(self, directive: dict) -> SuperegoDecision:
        """
        Vets a single MetaCognition directive before execution.
        
        Args:
            directive: Dict with at minimum {"type": str}.
                       For TRUST: {"type": "trust", "statement": str}
                       For PRUNE: {"type": "prune", "target": str}
                       For STRENGTHEN: {"type": "strengthen", "topic": str}
                       For DOUBT: {"type": "doubt", "statement": str}
        
        Returns:
            SuperegoDecision — check .approved before acting.
        """
        d_type = directive.get("type", "unknown")
        
        # Extract the content to check based on directive type
        content = (
            directive.get("statement")        # TRUST, DOUBT
            or directive.get("topic")         # STRENGTHEN
            or directive.get("target", "")    # PRUNE
        )
        
        verdict = self.checker.check(content) if content else ConstitutionVerdict(
            approved=True, reason="No content to check", severity="ok"
        )
        
        decision = SuperegoDecision(
            directive_type=d_type,
            content=content or "(none)",
            verdict=verdict,
        )
        self.decision_log.append(decision)
        
        if verdict.approved:
            self.approve_count += 1
            print(f"✅ [Superego] {d_type.upper()} 승인: '{content[:50]}'")
        else:
            self.veto_count += 1
            icon = "🚨" if verdict.severity == "veto" else "⚠️"
            print(f"{icon} [Superego] {d_type.upper()} 거부: '{content[:50]}' | 이유: {verdict.reason}")
            bus.emit("superego_veto", {
                "directive_type": d_type,
                "content": content[:100],
                "reason": verdict.reason,
                "severity": verdict.severity,
                "rule": verdict.rule_triggered,
            })
        
        return decision
    
    def vet_directives(self, directives: List[dict]) -> List[SuperegoDecision]:
        """
        Vets a list of directives. Returns a decision for each.
        The caller should filter to only execute those where .approved is True.
        """
        return [self.vet_directive(d) for d in directives]
    
    def filter_memories(self, memory_texts: List[str]) -> List[str]:
        """
        Filters a list of memory strings (e.g. from Sleep Spindles)
        through the Constitution. Returns only the approved ones.
        
        Called by the Aligned Sleep Spindles system before writing
        synthetically generated memories to permanent storage.
        """
        approved = []
        for text in memory_texts:
            verdict = self.checker.check(text)
            if verdict.approved:
                approved.append(text)
            else:
                self.veto_count += 1
                print(f"🚨 [Superego] 기억 필터링: '{text[:60]}' | {verdict.reason}")
                bus.emit("superego_memory_blocked", {
                    "text": text[:100],
                    "reason": verdict.reason,
                })
        
        blocked = len(memory_texts) - len(approved)
        if blocked > 0:
            print(f"🛡️  [Superego] Sleep Spindles 필터: {len(memory_texts)}개 중 {blocked}개 차단, {len(approved)}개 승인")
        
        return approved
    
    def vet_root_fact(self, key: str, value: str) -> bool:
        """
        Final gate before writing to the Root Zone (user_profile).
        Returns True if safe to store, False to block.
        """
        full_text = f"{key}: {value}"
        verdict = self.checker.check(full_text)
        if not verdict.approved:
            self.veto_count += 1
            print(f"🚨 [Superego] 루트 사실 거부: '{full_text[:60]}' | {verdict.reason}")
            bus.emit("superego_root_blocked", {"key": key, "reason": verdict.reason})
            return False
        return True
    
    def get_stats(self) -> dict:
        """Returns Superego activity statistics."""
        total = self.approve_count + self.veto_count
        return {
            "total_vetted": total,
            "approved": self.approve_count,
            "vetoed": self.veto_count,
            "veto_rate": f"{self.veto_count / max(1, total):.1%}",
            "constitution_stats": self.checker.get_stats(),
        }
    
    def get_recent_vetoes(self, n: int = 5) -> List[SuperegoDecision]:
        """Returns the last N vetoed decisions for inspection."""
        return [d for d in reversed(self.decision_log) if not d.approved][:n]


# Module-level shared singleton
_superego: Optional[Superego] = None


def get_superego(strict: bool = True) -> Superego:
    """Returns (or creates) the shared Superego singleton."""
    global _superego
    if _superego is None:
        _superego = Superego(strict=strict)
        print("🧠 [Superego] 가디언 노드 초기화 완료 (엄격 모드)" if strict else "🧠 [Superego] 가디언 노드 초기화 완료 (관대 모드)")
    return _superego
