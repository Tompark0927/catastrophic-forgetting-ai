"""
fza_self_architect.py — The Singularity Orchestrator (v14.0)
=============================================================
The top-level orchestrator that ties together:
  - KernelForge (bottleneck detection + auto-compilation)
  - LobeSpawner (autonomous sub-network creation)

This is the "cognitive executive" that watches FZA's own performance
and autonomously improves its architecture — without human intervention.

Responsibilities:
1. BOTTLENECK WATCH: monitors all key FZA inference paths via KernelForge.
   When a hot-path is detected, triggers auto-compilation.

2. DOMAIN GAP DETECTION: analyzes query patterns to detect when FZA
   is consistently failing in a new domain (unknown vocabulary, poor
   confidence, repeated DOUBT directives). When detected, triggers
   LobeSpawner to grow a new expert sub-network.

3. REFLECTION REPORT: generates a plaintext self-assessment of the 
   current architectural state — how many lobes exist, which kernels
   have been compiled, where bottlenecks remain.

Biological metaphor: The prefrontal cortex's executive control functions —
working memory management, task switching, and meta-cognition.
Combined with neuroplasticity: the ability to reshape one's own neural
circuitry in response to persistent cognitive demands.

Usage:
    from fza_self_architect import SelfArchitect
    sa = SelfArchitect(model=engine.model)
    
    # Watch an inference call
    result = sa.watched_inference(engine._generate, prompt)
    
    # Check for domain gaps after a conversation
    sa.analyze_domain_gaps(recent_directives=[...])
    
    # Get the full architectural self-assessment
    print(sa.reflect())
"""

import time
from typing import List, Optional
import torch
import torch.nn as nn

from fza_kernel_forge import KernelForge
from fza_lobe_spawner import LobeSpawner
from fza_event_bus import bus


# Minimum number of failures in one domain before we spawn a lobe for it
DOMAIN_GAP_THRESHOLD = 3

# Known domains FZA might need specialized lobes for
KNOWN_DOMAINS = [
    "quantum_physics", "music_theory", "legal_analysis", "medical_science",
    "financial_modeling", "advanced_mathematics", "software_architecture",
    "linguistics", "history", "molecular_biology",
]


class SelfArchitect:
    """
    The cognitive executive and self-improvement orchestrator.
    Monitors FZA's performance and autonomously rewrites its own architecture.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
        lobes_dir: str = "./lobes",
    ):
        self.model = model
        self.device = device
        
        # Initialize subsystems
        self.forge = KernelForge(auto_compile=True, device=device)
        self.spawner = LobeSpawner(model=model, lobes_dir=lobes_dir)
        
        # Domain gap tracker: domain → failure count
        self._domain_failures: dict = {}
        self._auto_spawned: List[str] = []   # Domains that were auto-spawned
        
        self.total_inferences_watched = 0
        self.started_at = time.time()
        
        print(f"🏛️  [SelfArchitect] 초기화 완료 — "
              f"로브: {self.spawner.get_stats()['total_lobes']}개, "
              f"커널 포지 준비 완료")
    
    # ── Kernel Forge Interface ────────────────────────────────────────────────
    
    def watched_inference(self, fn, *args, **kwargs):
        """
        Wraps any inference function with KernelForge profiling.
        On hot detection, triggers auto-compilation.
        
        Usage:
            result = sa.watched_inference(engine._generate, prompt)
        """
        self.total_inferences_watched += 1
        result = self.forge.profile("inference_generate", fn, *args, **kwargs)
        
        rec = self.forge._records.get("inference_generate")
        if rec and rec.is_hot and not rec.compiled:
            try:
                self.forge._try_compile("inference_generate", fn)
                bus.emit("kernel_compiled", {
                    "fn": "inference_generate",
                    "strategy": rec.compile_strategy,
                    "avg_ms": rec.avg_ms,
                })
            except Exception as e:
                print(f"⚠️  [SelfArchitect] 컴파일 실패: {e}")
        
        return result
    
    def watch_fn(self, name: str):
        """Decorator factory: wraps any function with forge profiling."""
        return self.forge.watch(name)
    
    # ── Domain Gap Detection ──────────────────────────────────────────────────
    
    def analyze_domain_gaps(self, recent_directives: List[dict] = None, query: str = ""):
        """
        Analyzes recent DOUBT directives and query patterns to detect
        whether FZA is consistently struggling with a new domain.
        
        If a domain registers >= DOMAIN_GAP_THRESHOLD failures and no lobe
        exists for it yet, autonomously spawns one.
        
        Args:
            recent_directives: List of directive dicts from MetaCognition
            query:             The current query string (for keyword scanning)
        """
        # Scan DOUBT directives for domain keywords
        if recent_directives:
            for d in recent_directives:
                if d.get("type") == "doubt":
                    domain = self._classify_domain(d.get("statement", ""))
                    if domain:
                        self._domain_failures[domain] = self._domain_failures.get(domain, 0) + 1
        
        # Scan query for domain signals
        if query:
            domain = self._classify_domain(query)
            if domain:
                # A single query signals potential need — not enough to spawn alone
                self._domain_failures[domain] = self._domain_failures.get(domain, 0) + 0.2
        
        # Check if any domain crossed the threshold
        for domain, failures in self._domain_failures.items():
            if failures >= DOMAIN_GAP_THRESHOLD:
                existing = self.spawner.get_by_domain(domain)
                if not existing:
                    self._auto_spawn_lobe(domain)
    
    def _auto_spawn_lobe(self, domain: str):
        """Autonomously spawns a lobe for a detected domain gap."""
        # Infer model hidden size (default to 4096 for Mistral-7B)
        hidden_dim = 4096
        if self.model:
            try:
                for p in self.model.parameters():
                    if len(p.shape) == 2 and p.shape[0] == p.shape[1]:
                        hidden_dim = p.shape[0]
                        break
            except Exception:
                pass
        
        lobe_id = self.spawner.spawn(
            domain=domain,
            input_dim=hidden_dim,
            hidden_dim=min(512, hidden_dim // 8),
            output_dim=hidden_dim,
        )
        self._auto_spawned.append(domain)
        
        bus.emit("lobe_spawned", {
            "domain": domain,
            "lobe_id": lobe_id,
            "trigger": "domain_gap_auto_detect",
        })
        print(f"🌱 [SelfArchitect] 도메인 갭 자동 감지 → '{domain}' 로브 자동 생성!")
    
    def _classify_domain(self, text: str) -> Optional[str]:
        """Simple keyword-based domain classifier."""
        text_lower = text.lower()
        domain_keywords = {
            "quantum_physics": ["quantum", "파동함수", "superposition", "엔탈피", "슈뢰딩거"],
            "music_theory": ["chord", "harmony", "음계", "음악 이론", "counterpoint", "cadence"],
            "legal_analysis": ["법률", "계약", "소송", "판례", "litigation", "statute", "juridical"],
            "medical_science": ["진단", "처방", "증상", "병리", "diagnosis", "pathology", "clinical"],
            "financial_modeling": ["dcf", "valuation", "주가", "포트폴리오", "sharpe", "beta"],
            "advanced_mathematics": ["미적분", "선형대수", "위상수학", "eigenvalue", "manifold"],
            "molecular_biology": ["dna", "rna", "단백질", "게놈", "mrna", "crispr", "염기"],
        }
        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return domain
        return None
    
    # ── Reflection / Self-Assessment ─────────────────────────────────────────
    
    def reflect(self) -> str:
        """
        Generates a full self-assessment report of the current architecture.
        Printed in response to the [REFLECT] directive.
        """
        forge_stats = self.forge.get_stats()
        spawner_stats = self.spawner.get_stats()
        uptime_h = (time.time() - self.started_at) / 3600
        
        lines = [
            "🏛️  [SelfArchitect] 아키텍처 자가 진단:",
            f"   가동 시간: {uptime_h:.1f}시간",
            f"   감시된 추론: {self.total_inferences_watched}회",
            "",
            "⚒️  KernelForge:",
            f"   모니터링 함수: {forge_stats['functions_monitored']}개",
            f"   핫-패스 감지: {forge_stats['hot_paths']}개",
            f"   자동 컴파일: {forge_stats['total_compilations']}개",
            f"   torch.compile: {'사용 가능' if forge_stats['torch_compile_available'] else '불가'}",
            f"   triton: {'사용 가능' if forge_stats['triton_available'] else '불가'}",
            "",
            "🧠 LobeSpawner:",
            f"   총 로브: {spawner_stats['total_lobes']}개",
            f"   활성 로브: {spawner_stats['active_lobes']}개",
            f"   도메인: {', '.join(spawner_stats['domains']) or '없음'}",
            f"   자동 생성: {', '.join(self._auto_spawned) or '없음'}",
            "",
            "🔍 도메인 갭 시그널:",
        ]
        for domain, count in self._domain_failures.items():
            bar = "█" * min(10, int(count))
            lines.append(f"   {domain:<25} {bar} ({count:.1f}/{DOMAIN_GAP_THRESHOLD})")
        
        if not self._domain_failures:
            lines.append("   (현재 도메인 갭 없음)")
        
        report = "\n".join(lines)
        print(report)
        return report
    
    def spawn_lobe(self, domain: str) -> str:
        """Manually spawns a lobe for a given domain (CLI command)."""
        lobe_id = self.spawner.spawn(domain)
        return lobe_id
    
    def get_stats(self) -> dict:
        return {
            **self.forge.get_stats(),
            **self.spawner.get_stats(),
            "total_inferences_watched": self.total_inferences_watched,
            "auto_spawned_domains": self._auto_spawned,
        }
