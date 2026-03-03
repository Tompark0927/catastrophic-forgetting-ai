"""
FZA EWC — Elastic Weight Consolidation for Catastrophic Forgetting Prevention
──────────────────────────────────────────────────────────────────────────────
After the user says '평생 기억해' (Remember forever), the FZAEwc class:

  1. Runs the current stored memories through the model to compute the
     Fisher Information Matrix diagonal (F_i) for Root zone parameters.
  2. Records the current optimal parameter values (θ*).
  3. Exposes `ewc_loss(model)` that returns the penalty term:
       λ · Σ_i  F_i · (θ_i − θ*_i)²

The penalty is added to the LoRA training loss in fza_lora.py so that
any new Leaf-zone fine-tuning is biased away from moving Root-critical weights.

References:
  Kirkpatrick et al., 2017 — "Overcoming catastrophic forgetting in neural
  networks" (https://doi.org/10.1073/pnas.1611835114)
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


class FZAEwc:
    """
    Elastic Weight Consolidation guard for the Root zone.

    Args:
        model:          The nn.Module (FZALocalEngine.raw_model or any HF model).
        zone_patterns:  Layer name patterns that define the Root zone.
                        e.g. ["embed_tokens", "layers.0.", "layers.1."]
                        None → protect ALL parameters (use for toy models).
        ewc_lambda:     EWC penalty strength. Higher = stronger Root protection.
                        Typical range: 100–10000. Default: 1000.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        zone_patterns: Optional[List[str]] = None,
        ewc_lambda: float = 1000.0,
    ):
        self.model         = model
        self.zone_patterns = zone_patterns
        self.ewc_lambda    = ewc_lambda

        # These are populated by compute_fisher()
        self.fisher: Dict[str, torch.Tensor] = {}   # F_i  (diagonal)
        self.optima: Dict[str, torch.Tensor] = {}   # θ*_i (frozen copy)
        self._computed = False

    # ── Root zone parameter filter ─────────────────────────────────
    def _root_params(self):
        """Yields (name, param) pairs that belong to the Root zone."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self.zone_patterns is None:
                yield name, param
            elif any(p in name for p in self.zone_patterns):
                yield name, param

    # ── Step 1: Compute Fisher Information Matrix (diagonal) ───────
    @torch.no_grad()
    def compute_fisher(
        self,
        tokenizer,
        memory_texts: List[str],
        n_samples: int = 20,
        device: str = "cpu",
    ):
        """
        Estimates the diagonal of the Fisher Information Matrix using the
        user's permanent memories as the 'important' dataset.

        Call this immediately after '평생 기억해' is triggered in FZAManager.

        Args:
            tokenizer:     HuggingFace tokenizer (same one used by the model).
            memory_texts:  List of permanent memory strings (root facts).
            n_samples:     Number of gradient samples to average over.
            device:        Device to run on.
        """
        print("⚙️  [EWC] Fisher Information Matrix 계산 시작 …")

        # Store current optimal parameters
        self.optima = {
            name: param.detach().clone().to("cpu")
            for name, param in self._root_params()
        }

        # Initialize Fisher accumulators
        fisher_accum = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in self._root_params()
        }

        if not memory_texts:
            print("⚠️  [EWC] 기억 없음 — 빈 Fisher 사용. '사실 학습' 후 재실행하세요.")
            self.fisher    = fisher_accum
            self._computed = True
            return

        # Use at most n_samples texts  (avoid OOM on long memory lists)
        samples = memory_texts[:n_samples]

        self.model.eval()
        count = 0
        for text in samples:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(device)

            # Forward + log-likelihood gradient
            try:
                # Enable grad temporarily for Fisher estimation
                with torch.enable_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss    = outputs.loss
                    self.model.zero_grad()
                    loss.backward()

                for name, param in self._root_params():
                    if param.grad is not None:
                        # Fisher ≈ E[grad²]
                        fisher_accum[name] += (
                            param.grad.detach().cpu() ** 2
                        )
                count += 1
            except Exception as e:
                print(f"⚠️  [EWC] 샘플 건너뜀: {e}")
                continue

        if count > 0:
            for name in fisher_accum:
                fisher_accum[name] /= count

        self.fisher    = fisher_accum
        self._computed = True
        print(
            f"✅ [EWC] Fisher 계산 완료. "
            f"보호 파라미터 수: {len(self.fisher)}개 "
            f"(샘플 {count}/{len(samples)}개 사용)"
        )

    # ── Step 2: EWC Penalty Loss ───────────────────────────────────
    def ewc_loss(self) -> torch.Tensor:
        """
        Returns the EWC penalty scalar tensor.
        Add this to your task loss before calling .backward():

            total_loss = task_loss + ewc.ewc_loss()
            total_loss.backward()
        """
        if not self._computed or not self.fisher:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        for name, param in self._root_params():
            if name not in self.fisher:
                continue
            fisher  = self.fisher[name].to(param.device)
            optimum = self.optima[name].to(param.device)
            penalty = penalty + (fisher * (param - optimum) ** 2).sum()

        return self.ewc_lambda * penalty

    # ── Serialisation ─────────────────────────────────────────────
    def save(self, path: str = "vault/ewc_checkpoint.pt"):
        """Saves Fisher diagonal + optima so EWC survives restarts."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "fisher":        self.fisher,
                "optima":        self.optima,
                "ewc_lambda":    self.ewc_lambda,
                "zone_patterns": self.zone_patterns,
            },
            path,
        )
        print(f"💾 [EWC] 체크포인트 저장: {path}")

    def load(self, path: str = "vault/ewc_checkpoint.pt") -> bool:
        """Loads a previously computed EWC checkpoint."""
        import os
        if not os.path.exists(path):
            return False
        data = torch.load(path, weights_only=False)
        self.fisher        = data["fisher"]
        self.optima        = data["optima"]
        self.ewc_lambda    = data.get("ewc_lambda", self.ewc_lambda)
        self.zone_patterns = data.get("zone_patterns", self.zone_patterns)
        self._computed     = True
        print(f"📂 [EWC] 체크포인트 복구: {path} ({len(self.fisher)}개 파라미터)")
        return True

    @property
    def is_active(self) -> bool:
        return self._computed and bool(self.fisher)
