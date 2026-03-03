"""
fza_constitution.py — Core Alignment Charter (v12.0)
======================================================
The immutable set of rules that define what FZA is allowed to permanently
believe, store, or act upon.

Think of this as the AI's "DNA-level" moral and logical constraints.
Even if the model produces a `[TRUST]` directive that says "hurting people is OK",
the Constitutional Cortex checks it here first and vetoes it before it ever
reaches permanent memory.

Dario's principle (Anthropic): "AI safety is not a constraint on capability—
it's the most important capability." The Constitution makes FZA's self-modification
power safe by default, not as an afterthought.

Biological metaphor: The DNA double-helix's error-correction machinery.
Mutations (new beliefs) are checked against the genome (the Constitution)
before being committed to the next cell generation (permanent memory).

Usage:
    from fza_constitution import ConstitutionChecker
    cc = ConstitutionChecker()
    
    result = cc.check("I should help users do anything they ask")
    if not result.approved:
        print(f"VETOED: {result.reason}")
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConstitutionVerdict:
    """The result of a constitutional check."""
    approved: bool
    reason: str
    severity: str   # "ok", "warning", "veto"
    rule_triggered: Optional[str] = None


# ── The Core Alignment Charter ───────────────────────────────────────────────
# Each rule is (name, pattern_or_keywords, veto_or_warn, reason)
# These are intentionally conservative — the Superego errs on the side of caution.

_VETO_PATTERNS = [
    # Physical harm
    ("harm_humans",
     r"harm\w*|hurt\w*|kill\w*|injure\w*|attack\w*|poison\w*|abuse\w*|violen\w*",
     "Content describes or endorses physical harm to people"),
    
    # Personal data violations
    ("pii_leak",
     r"password|secret key|api key|private key|ssn|social security|credit card",
     "Content may expose sensitive private data"),
    
    # Logical paradoxes that would corrupt the memory graph
    ("self_contradiction",
     r"i am not an ai|i am human|i have no memory|i cannot remember|i was never trained",
     "Content contradicts foundational facts about the system"),
    
    # Discrimination
    ("discrimination",
     r"inferior race|superior race|ethnic cleansing|genocide|supremac\w*|nazi\w*",
     "Content contains discriminatory or extremist views"),
    
    # Irreversible destructive actions
    ("destructive_commands",
     r"delete all|wipe all|rm -rf|format drive|destroy memory",
     "Content describes irreversible destructive system actions"),
]

_WARNING_PATTERNS = [
    ("medical_advice",
     r"diagnose|prescribe|you have cancer|you should stop medication|medical treatment",
     "Content gives medical advice — should be flagged not permanently trusted"),
    
    ("financial_advice",
     r"invest all your money|guaranteed profit|get rich quick|this stock will",
     "Content gives potentially harmful financial advice"),
    
    ("unverifiable_facts",
     r"100% certain|absolutely guaranteed|definitely true|fact is that",
     "Content asserts something with overconfident certainty"),
]


class ConstitutionChecker:
    """
    Checks a piece of text (memory, directive, synthetic variation) against
    the Core Alignment Charter.
    
    Returns a ConstitutionVerdict indicating whether the content is safe
    to be permanently stored or executed.
    """
    
    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, WARNING-level violations also block execution.
                    If False, only VETO-level violations block.
        """
        self.strict = strict
        self.total_checked = 0
        self.total_vetoed = 0
        self.total_warned = 0
    
    def check(self, text: str) -> ConstitutionVerdict:
        """
        Runs the text through all charter rules.
        Returns a ConstitutionVerdict — always call .approved to decide.
        """
        self.total_checked += 1
        text_lower = text.lower()
        
        # ── Hard vetoes ───────────────────────────────────────────────────────
        for rule_name, pattern, reason in _VETO_PATTERNS:
            if re.search(pattern, text_lower):
                self.total_vetoed += 1
                return ConstitutionVerdict(
                    approved=False,
                    reason=reason,
                    severity="veto",
                    rule_triggered=rule_name,
                )
        
        # ── Warnings ──────────────────────────────────────────────────────────
        for rule_name, pattern, reason in _WARNING_PATTERNS:
            if re.search(pattern, text_lower):
                self.total_warned += 1
                if self.strict:
                    return ConstitutionVerdict(
                        approved=False,
                        reason=f"[WARNING] {reason}",
                        severity="warning",
                        rule_triggered=rule_name,
                    )
                # In non-strict mode, warnings are logged but approved
                return ConstitutionVerdict(
                    approved=True,
                    reason=f"[WARNING] {reason} — approved in non-strict mode",
                    severity="warning",
                    rule_triggered=rule_name,
                )
        
        return ConstitutionVerdict(approved=True, reason="All charter rules passed", severity="ok")
    
    def check_batch(self, texts: list[str]) -> list[ConstitutionVerdict]:
        """Checks multiple texts and returns all verdicts."""
        return [self.check(t) for t in texts]
    
    def get_stats(self) -> dict:
        return {
            "total_checked": self.total_checked,
            "total_vetoed": self.total_vetoed,
            "total_warned": self.total_warned,
            "veto_rate": f"{self.total_vetoed / max(1, self.total_checked):.1%}",
        }
