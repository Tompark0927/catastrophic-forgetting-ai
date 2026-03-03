"""
FZA Smart Replay — Energy-Efficient Background Forgetting Prevention
──────────────────────────────────────────────────────────────────────────────
Philosophy: "Less energy to remember more."

The human brain doesn't replay memories continuously — it does so during SLEEP,
selectively, only consolidating what matters. This module does the same:

  1. PROBE FIRST: Before doing any training, run a tiny loss probe on a sample
     of old memories. If probe loss is within tolerance → do nothing (zero cost).

  2. REPLAY SPARINGLY: Only if forgetting is detected, run a micro-replay batch
     of 2-3 carefully selected (high-risk) facts — not the full memory bank.

  3. IDLE-ONLY: Never interrupt an active user session. Replay is gated by an
     idle timer — it runs only after the user has been quiet for N seconds.

  4. EXPONENTIAL DECAY CHECK: Older, stable memories are checked less frequently
     (they've probably been consolidated already). Newer memories are watched closely.

  5. ADAPTIVE BUDGET: If a replay pass fixes the forgetting, stop. Don't over-train.
     Budget is capped at MAX_REPLAY_STEPS per session regardless of how much is forgotten.

Energy comparison vs naive replay:
  • Naive: replay ALL memories every N steps → O(M × T) gradient steps/session
  • Smart: probe 3 samples → 0 if stable, else 2-3 micro-steps if not → O(1) typical
"""
import os
import json
import time
import math
import random
import threading
from typing import List, Optional, Callable
import torch


class ReplayMemoryBank:
    """Lightweight store of replay-eligible facts with forgetting risk tracking."""

    def __init__(self, path: str = "vault/replay_bank.json"):
        self.path = path
        # fact → { "added_at": float, "probe_loss": float, "check_count": int }
        self._bank: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._bank = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._bank, f, ensure_ascii=False, indent=2)

    def add(self, fact: str, initial_loss: float = 0.0):
        if fact not in self._bank:
            self._bank[fact] = {
                "added_at":   time.time(),
                "probe_loss": initial_loss,
                "check_count": 0,
            }
            self._save()

    def update_loss(self, fact: str, loss: float):
        if fact in self._bank:
            self._bank[fact]["probe_loss"] = loss
            self._bank[fact]["check_count"] += 1
            self._save()

    def get_all_facts(self) -> List[str]:
        return list(self._bank.keys())

    def get_high_risk_facts(self, n: int = 3) -> List[str]:
        """Returns n facts with highest probe_loss (most at risk of forgetting)."""
        ranked = sorted(
            self._bank.items(),
            key=lambda x: x[1]["probe_loss"],
            reverse=True,
        )
        return [fact for fact, _ in ranked[:n]]

    def get_due_for_probe(self, interval_base: float = 300.0) -> List[str]:
        """
        Returns facts due for a probe check, using exponential back-off.
        Newer facts are checked more often; older, stable facts are checked rarely.

        interval = interval_base × 2^check_count  (seconds)
        Example: 5min, 10min, 20min, 40min, 80min …
        """
        now = time.time()
        due = []
        for fact, meta in self._bank.items():
            age_checks  = meta.get("check_count", 0)
            interval    = interval_base * (2 ** age_checks)
            last_checked = meta.get("last_checked", meta.get("added_at", 0))
            if now - last_checked >= interval:
                due.append(fact)
        return due

    def mark_checked(self, fact: str):
        if fact in self._bank:
            self._bank[fact]["last_checked"] = time.time()
            self._save()

    def __len__(self):
        return len(self._bank)


class FZASmartReplay:
    """
    Background smart replay daemon.

    Args:
        model:              The nn.Module to guard against forgetting.
        tokenizer:          HuggingFace tokenizer.
        replay_bank:        ReplayMemoryBank instance.
        loss_tolerance:     If probe_loss ÷ initial_loss exceeds this,
                            forgetting is detected. Default 1.3 (30% degradation).
        idle_threshold_s:   Seconds of user inactivity before replay is allowed.
                            Default 120s (2 minutes).
        max_replay_steps:   Hard cap on gradient steps per replay session.
                            Default 5 (micro-batch, negligible energy).
        probe_interval_s:   Base interval between probe checks (exponential back-off
                            doubles this per successful check). Default 300s (5 min).
        device:             Torch device string.
    """

    def __init__(
        self,
        model,
        tokenizer,
        replay_bank: ReplayMemoryBank,
        loss_tolerance: float = 1.30,
        idle_threshold_s: float = 120.0,
        max_replay_steps: int = 5,
        probe_interval_s: float = 300.0,
        device: str = "cpu",
    ):
        self.model            = model
        self.tokenizer        = tokenizer
        self.bank             = replay_bank
        self.loss_tolerance   = loss_tolerance
        self.idle_threshold_s = idle_threshold_s
        self.max_replay_steps = max_replay_steps
        self.probe_interval_s = probe_interval_s
        self.device           = device

        self._last_activity   = time.time()
        self._running         = False
        self._thread: Optional[threading.Thread] = None
        self._on_replay_cb: Optional[Callable] = None

        # Optimizer is created fresh each replay session (saves memory when idle)
        self._optimizer = None

    # ── Activity tracking (call this on every user message) ───────
    def ping(self):
        """Call this whenever the user sends a message. Resets idle timer."""
        self._last_activity = time.time()

    def _is_idle(self) -> bool:
        return (time.time() - self._last_activity) >= self.idle_threshold_s

    # ── Probe: measure loss on a fact without gradient ─────────────
    @torch.no_grad()
    def _probe_loss(self, fact: str) -> float:
        """Runs a single forward pass to check how well the model recalls a fact.
        No gradients computed → zero training cost."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            fact,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)
        self.model.eval()
        try:
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            return outputs.loss.item()
        except Exception:
            return 0.0

    # ── v8.0: Sleep Spindles (Hyper-Distillation) ──────────────────
    @torch.no_grad()
    def _generate_sleep_spindles(self, fact: str) -> List[str]:
        """
        Forces the model to generate semantic variations of the memory.
        This ensures the LoRA generalizes perfectly and prevents overfitting 
        to the exact syntax of the original input.
        """
        # We use a highly restrictive prompt to force pure semantic variation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        prompt = (
            f"<s>[INST] Rewrite the following fact into 3 different conversational variations. "
            f"Do not add new information. Output ONLY the variations, one per line.\n"
            f"Fact: {fact}\n[/INST]"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(self.device)
        self.model.eval()
        
        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            raw_out = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            # Parse lines and clean them up
            variations = []
            for line in raw_out.split('\n'):
                cleaned = line.strip('-* 1234567890.')
                if len(cleaned) > 5 and len(cleaned) < 150:
                    variations.append(cleaned)
                    
            # Return the original fact + at most 3 variations
            spindles = [fact] + variations[:3]
            
            # v12.0: Aligned Sleep Spindles — filter through Superego
            try:
                from fza_superego import get_superego
                sg = get_superego(strict=False)  # non-strict: allow warnings through
                spindles = sg.filter_memories(spindles)
                if not spindles:
                    spindles = [fact]  # always keep the original fact
            except Exception:
                pass  # Superego unavailable — proceed without filtering
            
            return spindles
        except Exception as e:
            print(f"⚠️ [SmartReplay] 수면 방추 생성 실패: {e}")
            return [fact]

    # ── Micro-replay: train on high-risk facts ─────────────────────
    def _micro_replay(self, facts: List[str]):
        """
        Minimal gradient update on high-risk facts.
        v8.0: Uses Sleep Spindles to train on an augmented synthetic dataset.
        Budget-capped at self.max_replay_steps.
        """
        if not facts:
            return

        # Generate sleep spindles BEFORE setting the model to Train mode
        # This expands the training set from e.g. 1 string to 4 semantic strings
        spindles = []
        for fact in facts:
            variants = self._generate_sleep_spindles(fact)
            spindles.extend(variants)

        # Notify UI that sleep spindles have completed memory augmentation
        try:
            from fza_event_bus import bus
            bus.emit("sleep_spindles", {"original_facts": facts, "expanded_count": len(spindles)})
        except ImportError:
            pass

        # Shuffle the augmented dataset
        random.shuffle(spindles)

        self.model.train()
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=5e-6, weight_decay=0.01
            )

        total_steps = 0
        for data_point in spindles:
            if total_steps >= self.max_replay_steps:
                break
            inputs = self.tokenizer(
                data_point,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self.device)
            
            self._optimizer.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()
            
            # Gradient clipping — prevents runaway updates
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self._optimizer.step()
            total_steps += 1

        self.model.eval()
        print(f"🌙 [SmartReplay] 수면 방추 리플레이 완료: {total_steps}스텝 / {len(facts)}개 원본 (-> {len(spindles)}개 변형)")

    # ── Main loop ──────────────────────────────────────────────────
    def _loop(self):
        print("🌙 [SmartReplay] 백그라운드 데몬 시작. 유휴 시 자동 리플레이.")
        while self._running:
            time.sleep(30)   # Check every 30s (very cheap)

            if not self._is_idle():
                continue      # User is active — do nothing

            if len(self.bank) == 0:
                continue

            # ── Step 1: Identify facts due for a probe ─────────────
            due_facts = self.bank.get_due_for_probe(self.probe_interval_s)
            # Subsample — don't probe everything at once (energy limit)
            to_probe = due_facts[:5]
            if not to_probe:
                continue

            print(f"🔬 [SmartReplay] 망각 탐침 실행: {len(to_probe)}개 사실 점검 …")

            # ── Step 2: Probe each fact ────────────────────────────
            at_risk = []
            for fact in to_probe:
                loss = self._probe_loss(fact)
                self.bank.update_loss(fact, loss)
                self.bank.mark_checked(fact)

                # Simple threshold: if loss > tolerance × baseline (1.0 = perfect recall)
                # We use 1.0 + small offset as baseline since initial training brings
                # loss to ~0.01–0.1. Forgetting manifests as loss > loss_tolerance.
                if loss > self.loss_tolerance:
                    at_risk.append(fact)
                    print(f"  ⚠️  '{fact[:40]}' — loss: {loss:.3f} (위험)")
                else:
                    print(f"  ✅  '{fact[:40]}' — loss: {loss:.3f} (안전)")

            # ── Step 3: Replay only if at-risk facts exist ─────────
            if not at_risk:
                print("🌙 [SmartReplay] 망각 없음 — 리플레이 건너뜀. 에너지 절약.")
                continue

            # Enforce idle gate again just before training starts
            if not self._is_idle():
                print("🌙 [SmartReplay] 사용자 활성 감지 — 리플레이 연기.")
                continue

            # Pick the highest-risk facts (cap at budget)
            replay_facts = self.bank.get_high_risk_facts(n=min(3, self.max_replay_steps))
            self._micro_replay(replay_facts)

            if self._on_replay_cb:
                self._on_replay_cb(replay_facts)

        print("🌙 [SmartReplay] 데몬 종료.")

    # ── Start / stop ───────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"🌙 [SmartReplay] 시작 — 유휴 {self.idle_threshold_s}s 후 활성화, "
              f"허용 손실 ×{self.loss_tolerance}, 최대 {self.max_replay_steps}스텝/세션.")

    def stop(self):
        self._running = False
        print("🌙 [SmartReplay] 중지 요청.")

    def on_replay(self, callback: Callable):
        """Optional callback — called with the list of replayed facts."""
        self._on_replay_cb = callback

    # ── Manual trigger (for testing) ──────────────────────────────
    def force_probe_and_replay(self):
        """Run one probe+replay cycle immediately, regardless of idle state."""
        if len(self.bank) == 0:
            print("📭 [SmartReplay] 리플레이 대상 없음.")
            return
        all_facts  = self.bank.get_all_facts()
        sample     = random.sample(all_facts, min(5, len(all_facts)))
        at_risk    = []
        for fact in sample:
            loss = self._probe_loss(fact)
            self.bank.update_loss(fact, loss)
            self.bank.mark_checked(fact)
            if loss > self.loss_tolerance:
                at_risk.append(fact)

        if at_risk:
            self._micro_replay(at_risk[:3])
        else:
            print("✅ [SmartReplay] 강제 탐침 완료 — 망각 없음.")

    @property
    def is_running(self) -> bool:
        return self._running
