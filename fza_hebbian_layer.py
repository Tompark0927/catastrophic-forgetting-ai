"""
FZA Hebbian Layer — Zero-Backprop Fast-Weight Associative Memory
──────────────────────────────────────────────────────────────────────────────
This is the core of FZA v5.0 / NFC (Neuromorphic Fast-Weight Consolidation).

It implements the Hebbian learning rule inside a custom torch.nn.Module that
can be injected into any HuggingFace Transformer's hidden state pipeline.

                    ┌─────────────────────────────────────┐
  hidden_state h ──►│ W_fast × h  (outer product update)  │──► augmented h'
                    │ Updates in ONE forward pass, ZERO    │
                    │ gradient. Saturates at capacity.    │
                    └─────────────────────────────────────┘

Biological Basis:
  In the hippocampus (CA3/CA1 regions), Hebbian synaptic potentiation
  (LTP — Long-Term Potentiation) causes a synapse to strengthen when
  "neurons that fire together, wire together." One exposure can create a
  synapse strong enough to drive recall.

  This module implements the same principle using a Fast Weight matrix:
    W ← W + η · (target ⊗ query)
  where ⊗ is the outer product of the output (value) and input (key) vectors.
  No loss function. No backward pass. No optimizer.

  The layer is finite in capacity (dim² parameters). It degrades
  gracefully as it approaches saturation (signal-to-noise decreases).
  The FZA Smart Replay daemon detects saturation via the probe mechanism
  and distills the layer into a permanent frozen LoRA adapter,
  then calls flush() to wipe it clean for the next day.

Usage (standalone):
    layer = FZAHebbianLayer(hidden_dim=4096)
    # Register a new fact (one forward pass, no backprop):
    layer.hebbian_update(query_hidden, value_hidden, lr=0.01)
    # At inference, the layer automatically adds fast-weight recall:
    h_enriched = layer(h)

Usage (injected into LocalEngine):
    engine = FZALocalEngine(...)
    layer  = FZAHebbianLayer.inject_into(engine.raw_model, layer_index=-1)
    layer.hebbian_update(...)  # facts are now immediately encoded
"""
import os
import math
import torch
import torch.nn as nn
from typing import Optional


class FZAHebbianLayer(nn.Module):
    """
    A zero-backprop associative fast-weight layer.

    The weight matrix W maps query vectors (keys) to output vectors (values),
    updated instantly via the outer product rule with no gradient computation.

    Parameters:
        hidden_dim:    Dimensionality of the transformer hidden state.
        lr:            Default Hebbian learning rate (δ per datum).
        max_norm:      L2 norm ceiling on W — prevents unbounded growth.
        decay:         Per-step weight decay (simulates synaptic forgetting
                       in the transient hippocampal buffer — keeps capacity fresh).
        device:        Torch device string.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        lr:         float = 0.01,
        max_norm:   float = 10.0,
        decay:      float = 0.999,
        device:     str   = "cpu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lr         = lr
        self.max_norm   = max_norm
        self.decay      = decay
        self.device     = device

        # Fast weight matrix W — starts at zero (empty hippocampus)
        # Not a nn.Parameter so it won't appear in optimizer.parameters()
        # and won't receive any gradients.
        self._W = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32, device=device)

        # Tracks how many Hebbian updates have been applied (saturation indicator)
        self._update_count: int = 0

        # Scalar gain for fast-weight contribution (anneals as W saturates)
        self._gate = nn.Parameter(torch.tensor(1.0))

    # ─── Hebbian Update (the "instant memory write") ─────────────────
    @torch.no_grad()
    def hebbian_update(
        self,
        query:  torch.Tensor,
        value:  torch.Tensor,
        lr:     Optional[float] = None,
    ):
        """
        Register a new associative binding (query → value) via outer product.

        Zero gradient computation. Executes in microseconds on any hardware.

        Args:
            query:  The "key" representation — what activates this memory.
                    Shape: (hidden_dim,) or (seq_len, hidden_dim)
            value:  The "value" representation — what should be recalled.
                    Shape: (hidden_dim,) or (seq_len, hidden_dim)
            lr:     Override the layer's default learning rate for this update.

        Math:
            W ← decay · W + η · mean(value) ⊗ mean(query)
        """
        eta = lr if lr is not None else self.lr

        # Pool sequences to single vectors if needed
        q = query.mean(dim=0) if query.dim() > 1 else query
        v = value.mean(dim=0) if value.dim() > 1 else value
        q = q.to(self.device, dtype=torch.float32)
        v = v.to(self.device, dtype=torch.float32)

        # Normalise (unit vectors → bounded outer product)
        q_norm = q / (torch.norm(q) + 1e-8)
        v_norm = v / (torch.norm(v) + 1e-8)

        # Outer product: (hidden_dim, 1) × (1, hidden_dim) → (hidden_dim, hidden_dim)
        delta_W = torch.outer(v_norm, q_norm)

        # Apply decay (simulates synaptic fading — keeps buffer capacity fresh)
        self._W.mul_(self.decay)

        # Add the new association
        self._W.add_(eta * delta_W)

        # Clamp to prevent W norm from blowing up
        w_norm = torch.norm(self._W)
        if w_norm > self.max_norm:
            self._W.mul_(self.max_norm / w_norm)

        self._update_count += 1

    # ─── Forward pass (auto-recall enhancement) ─────────────────────
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Enhances the hidden state stream using fast-weight recall.

        At each position, the fast-weight matrix acts as an associative
        memory overlay: if hidden_state is similar to a registered query,
        the corresponding stored value is added to the output.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            Augmented hidden states (same shape).
        """
        if self._update_count == 0:
            return hidden_states    # Short-circuit: no memories yet

        h = hidden_states.to(self.device, dtype=torch.float32)
        # W: (hidden_dim, hidden_dim)  h: (B, T, D) → reshape for batched matmul
        recall = torch.matmul(h, self._W.t())   # (B, T, D)

        # Gated addition: gate starts at 1.0, learned to suppress if harmful
        gate   = torch.sigmoid(self._gate)
        output = hidden_states + gate * recall.to(hidden_states.dtype)
        return output

    # ─── Capacity / Saturation metrics ──────────────────────────────
    @property
    def saturation(self) -> float:
        """
        Saturation ∈ [0, 1]: how full the fast-weight buffer is.
        Defined as (current W Frobenius norm) / max_norm.
        Near 1.0 → buffer is full, distillation should be triggered.
        """
        return float(torch.norm(self._W).item() / self.max_norm)

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def is_saturated(self) -> bool:
        return self.saturation > 0.85

    # ─── Flush (called after distillation into stable LoRA) ──────────
    @torch.no_grad()
    def flush(self):
        """
        Wipe the fast-weight matrix clean (sleep consolidation complete).
        Equivalent to the hippocampus clearing itself after dreaming.
        """
        self._W.zero_()
        self._update_count = 0
        print("🌊 [HebbianLayer] 패스트 웨이트 초기화 완료 — 새 사실을 받을 준비 완료.")

    # ─── Serialisation ───────────────────────────────────────────────
    def save(self, path: str = "vault/hebbian_layer.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "W":            self._W,
            "update_count": self._update_count,
            "hidden_dim":   self.hidden_dim,
            "lr":           self.lr,
            "max_norm":     self.max_norm,
            "decay":        self.decay,
        }, path)
        print(f"💾 [HebbianLayer] 저장 완료: {path} (포화도: {self.saturation:.1%})")

    def load(self, path: str = "vault/hebbian_layer.pt") -> bool:
        if not os.path.exists(path):
            return False
        data = torch.load(path, weights_only=False, map_location=self.device)
        self._W            = data["W"].to(self.device)
        self._update_count = data["update_count"]
        self.hidden_dim    = data["hidden_dim"]
        self.lr            = data["lr"]
        self.max_norm      = data["max_norm"]
        self.decay         = data["decay"]
        print(f"📂 [HebbianLayer] 복구: {path} (업데이트 {self._update_count}회, 포화도 {self.saturation:.1%})")
        return True

    # ─── Static factory: inject into a HuggingFace model ─────────────
    @staticmethod
    def inject_into(
        model,
        layer_index: int = -1,
        hidden_dim: Optional[int] = None,
        **kwargs,
    ) -> "FZAHebbianLayer":
        """
        Injects the Hebbian layer as a post-hook on the specified transformer
        block's hidden state output.

        layer_index: which transformer layer to hook into.
                     -1 = last layer (highest-level semantic representations).

        Returns the FZAHebbianLayer instance (for hebbian_update() calls).
        """
        # Auto-detect hidden dimension
        if hidden_dim is None:
            cfg = getattr(model, "config", None)
            hidden_dim = (
                getattr(cfg, "hidden_size", None) or
                getattr(cfg, "d_model", None) or
                4096
            )

        device = str(next(model.parameters()).device)
        layer  = FZAHebbianLayer(hidden_dim=hidden_dim, device=device, **kwargs)

        # Identify the target transformer block
        blocks = None
        for attr in ("layers", "model.layers", "transformer.h", "model.decoder.layers"):
            try:
                obj = model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "__len__") and len(obj) > 0:
                    blocks = obj
                    break
            except AttributeError:
                continue

        if blocks is None:
            print("⚠️  [HebbianLayer] 트랜스포머 블록을 찾지 못해 훅 삽입 실패.")
            return layer

        target_block = blocks[layer_index % len(blocks)]

        def _hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            h_aug = layer(h)
            if isinstance(output, tuple):
                return (h_aug,) + output[1:]
            return h_aug

        target_block.register_forward_hook(_hook)
        print(f"🧬 [HebbianLayer] Layer {layer_index} 훅 삽입 완료. "
              f"히든 차원: {hidden_dim}. 즉각 학습 준비 완료.")
        return layer

    def __repr__(self):
        return (f"FZAHebbianLayer(dim={self.hidden_dim}, "
                f"updates={self._update_count}, sat={self.saturation:.1%})")
