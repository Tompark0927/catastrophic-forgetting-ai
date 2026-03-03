"""
fza_meta_cognition.py — Zero-Shot Self-Modification (v10.0)
============================================================
"The AI gains the ability to edit its own Associative Memory Graph.
It can deliberately prune connections it deems noisy or explicitly
strengthen nodes that represent core logical truths — effectively
achieving meta-cognition."
— FZA Master Roadmap

How it works:
  1. After generating each reply, the MetaCognition engine scans
     the model's output for special self-directive tokens:
     
       [PRUNE: <adapter_id>]   → removes a noisy memory adapter
       [STRENGTHEN: <topic>]   → reinforces graph edge weights for a topic
       [TRUST: <statement>]    → permanently encodes a fact as a root truth
       [DOUBT: <statement>]    → marks a fact as uncertain / lowers confidence
       [REFLECT]               → triggers a self-assessment of memory health
     
  2. These directives are stripped from the final output before
     it reaches the user (invisible to the user unless you want to show them).
     
  3. The directives are executed on the live FZA state:
     Memory Graph edges, adapter weights, and root fact confidence scores.

Biological metaphor: Metacognition in humans — the prefrontal cortex
monitoring and editing its own beliefs. This is the closest thing to
genuine self-awareness in an AI system.

Usage:
    from fza_meta_cognition import MetaCognitionEngine
    mc = MetaCognitionEngine(manager=fza_manager)
    
    raw_output = "좋은 질문이야. [REFLECT] 나는 Tom에 대해 잘 알고 있어."
    clean_output, directives = mc.process(raw_output)
    # clean_output → "좋은 질문이야. 나는 Tom에 대해 잘 알고 있어."
    # directives   → [{"type": "reflect"}]
"""

import re
import time
from typing import Optional
from fza_event_bus import bus


# Parsing patterns for self-directive tokens
_PATTERNS = {
    "prune":      re.compile(r'\[PRUNE:\s*([^\]]+)\]'),
    "strengthen": re.compile(r'\[STRENGTHEN:\s*([^\]]+)\]'),
    "trust":      re.compile(r'\[TRUST:\s*([^\]]+)\]'),
    "doubt":      re.compile(r'\[DOUBT:\s*([^\]]+)\]'),
    "reflect":    re.compile(r'\[REFLECT\]'),
}


class MetaCognitionEngine:
    """
    Scans model output for self-modification directives and executes them.
    """
    
    def __init__(self, manager):
        """
        Args:
            manager: FZAManager instance — gives access to Memory Graph,
                     adapter bank, root facts, and EWC state.
        """
        self.manager = manager
        self.directive_log: list[dict] = []
        self.total_directives = 0
    
    def process(self, raw_output: str) -> tuple[str, list[dict]]:
        """
        Parses, strips, and executes all self-directive tokens from the model's output.
        
        Returns:
            clean_output:  The output with all directive tokens removed (user sees this)
            directives:    List of directives that were found and executed
        """
        clean = raw_output
        executed = []
        
        # ── PRUNE ──────────────────────────────────────────────────────────────
        for m in _PATTERNS["prune"].finditer(raw_output):
            adapter_id = m.group(1).strip()
            result = self._exec_prune(adapter_id)
            executed.append({"type": "prune", "target": adapter_id, "result": result})
        clean = _PATTERNS["prune"].sub("", clean)
        
        # ── STRENGTHEN ─────────────────────────────────────────────────────────
        for m in _PATTERNS["strengthen"].finditer(raw_output):
            topic = m.group(1).strip()
            result = self._exec_strengthen(topic)
            executed.append({"type": "strengthen", "topic": topic, "result": result})
        clean = _PATTERNS["strengthen"].sub("", clean)
        
        # ── TRUST ──────────────────────────────────────────────────────────────
        for m in _PATTERNS["trust"].finditer(raw_output):
            statement = m.group(1).strip()
            result = self._exec_trust(statement)
            executed.append({"type": "trust", "statement": statement, "result": result})
        clean = _PATTERNS["trust"].sub("", clean)
        
        # ── DOUBT ──────────────────────────────────────────────────────────────
        for m in _PATTERNS["doubt"].finditer(raw_output):
            statement = m.group(1).strip()
            result = self._exec_doubt(statement)
            executed.append({"type": "doubt", "statement": statement, "result": result})
        clean = _PATTERNS["doubt"].sub("", clean)
        
        # ── REFLECT ────────────────────────────────────────────────────────────
        if _PATTERNS["reflect"].search(raw_output):
            result = self._exec_reflect()
            executed.append({"type": "reflect", "result": result})
        clean = _PATTERNS["reflect"].sub("", clean)
        
        # Normalize whitespace
        clean = " ".join(clean.split())
        
        if executed:
            self.total_directives += len(executed)
            self.directive_log.extend(executed)
            bus.emit("meta_cognition", {
                "directives": executed,
                "total_executed": self.total_directives,
            })
        
        return clean, executed
    
    # ── Directive Executors ──────────────────────────────────────────────────
    
    def _exec_prune(self, adapter_id: str) -> str:
        """Removes a LoRA adapter from the memory graph (prunes a noisy memory)."""
        try:
            if self.manager.router and hasattr(self.manager.router, 'bank'):
                bank = self.manager.router.bank
                if hasattr(bank, 'remove') and bank.exists(adapter_id):
                    bank.remove(adapter_id)
                    print(f"✂️ [MetaCog] PRUNE 실행 → 어댑터 {adapter_id[:8]} 제거됨")
                    return "pruned"
                else:
                    print(f"⚠️ [MetaCog] PRUNE: 어댑터 {adapter_id[:8]} 미발견")
                    return "not_found"
        except Exception as e:
            return f"error: {e}"
        return "no_router"
    
    def _exec_strengthen(self, topic: str) -> str:
        """
        Strengthens Memory Graph edge weights related to a topic.
        Finds the top-k adapters matching the topic and boosts their scores.
        """
        try:
            if self.manager.memory_graph:
                neighbors = self.manager.memory_graph.expand_neighbors(topic, top_k=3)
                if neighbors:
                    ids = [n[0] for n in neighbors]
                    print(f"💪 [MetaCog] STRENGTHEN → 주제 '{topic}': {len(ids)}개 노드 강화")
                    # Boost: re-visit these adapters so the graph scores them higher
                    for nid in ids:
                        self.manager.memory_graph.visit(nid)
                    return f"strengthened {len(ids)} nodes"
        except Exception as e:
            return f"error: {e}"
        return "no_graph"
    
    def _exec_trust(self, statement: str) -> str:
        """
        Permanently encodes a fact as an immutable root truth.
        Adds it directly to the Root Zone (user_profile).
        """
        try:
            if self.manager:
                key = f"trust_{int(time.time()) % 10000}"
                self.manager.user_profile[key] = statement
                self.manager._save_profile()
                print(f"🔒 [MetaCog] TRUST → 루트 사실 등록: '{statement[:50]}'")
                return "stored"
        except Exception as e:
            return f"error: {e}"
        return "no_manager"
    
    def _exec_doubt(self, statement: str) -> str:
        """
        Marks a statement as uncertain by logging it to a doubt register.
        In production, this would lower the EWC Fisher diagonal for 
        parameters associated with the uncertain statement.
        """
        print(f"🔍 [MetaCog] DOUBT → 불확실 등록: '{statement[:50]}'")
        doubt_record = {
            "statement": statement,
            "timestamp": time.time(),
        }
        # Store in a doubt register on the manager if available
        if not hasattr(self.manager, '_doubt_register'):
            self.manager._doubt_register = []
        self.manager._doubt_register.append(doubt_record)
        return "logged"
    
    def _exec_reflect(self) -> str:
        """
        Triggers a self-assessment of memory health. Prints a summary of
        the current memory state: adapters, root facts, EWC tension, etc.
        """
        print("🪞 [MetaCog] REFLECT 실행 — 기억 건강 보고서:")
        try:
            if self.manager:
                root_count = len([k for k in self.manager.user_profile if not k.startswith("_")])
                adapter_count = len(self.manager.router.bank.get_all_ids()) if self.manager.router else 0
                ewc_status = "활성화" if (self.manager.ewc and self.manager.ewc.is_active) else "비활성화"
                
                print(f"   루트 사실: {root_count}개")
                print(f"   어댑터 수: {adapter_count}개")
                print(f"   EWC 상태: {ewc_status}")
                print(f"   총 메타인지 지시: {self.total_directives}회")
                
                return f"root={root_count}, adapters={adapter_count}, ewc={ewc_status}"
        except Exception as e:
            return f"error: {e}"
        return "ok"
    
    def get_log(self, last_n: int = 10) -> list[dict]:
        """Returns the last N directive execution records."""
        return self.directive_log[-last_n:]
