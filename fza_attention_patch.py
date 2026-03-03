"""
fza_attention_patch.py — Native Hebbian Fast-Weights in Attention Blocks (v10.0)
==================================================================================
"Instead of attaching standard Feed-Forward Networks, we replace specific
attention blocks with native, differentiable Hebbian Fast-Weight Layers."
— FZA Master Roadmap

This module monkey-patches selected transformer attention modules at runtime,
injecting a lightweight FZAHebbianLayer alongside each attention block's output.
The result: the model's internal state morphs with every forward pass — no
backpropagation, no optimizer steps, no retraining.

Biological metaphor: Long-Term Potentiation (LTP) — synaptic connections
strengthen or weaken during use, not during dedicated sleep/replay sessions.

The patch is:
  AttentionBlock(hidden_state) → original_attn(hidden_state) + hebbian_delta(hidden_state)

The Hebbian delta is a zero-backprop outer product update applied to a small
Fast-Weight matrix F ∈ R^(d_model × d_model):
  F ← (1 - η) * F + η * (post ⊗ pre)  (Oja decay for stability)
  delta = F @ hidden_state
  output = original_output + scale * delta
"""

import torch
import torch.nn as nn
from typing import Optional


class AttentionHebbianPatch(nn.Module):
    """
    A minimal Fast-Weight module injected into a transformer attention block.
    Wraps any attention module and adds a Hebbian delta to the output.
    
    No gradient flows through the Hebbian update — it's an online, continuous
    update rule running in the model's forward pass itself.
    """
    
    def __init__(self, original_attn: nn.Module, d_model: int, eta: float = 0.01, scale: float = 0.05):
        """
        Args:
            original_attn: The attention module being wrapped
            d_model:       Hidden dimension size of the model
            eta:           Hebbian learning rate (default: 0.01, very gentle)
            scale:         How much the Hebbian delta affects the output (default: 0.05)
        """
        super().__init__()
        self.original_attn = original_attn
        self.d_model = d_model
        self.eta = eta
        self.scale = scale
        self.update_count = 0
        
        # The Fast-Weight matrix — never in gradient graph, always in-place updates
        self.register_buffer('fast_weight', torch.zeros(d_model, d_model))
    
    def forward(self, hidden_states, *args, **kwargs):
        """
        Forward pass:
        1. Run original attention
        2. Apply Hebbian Fast-Weight delta to output
        3. Update Fast-Weight matrix using Oja's rule (no grad)
        """
        # Run the original attention
        attn_output = self.original_attn(hidden_states, *args, **kwargs)
        
        # Extract tensor if output is a tuple (as is common in HuggingFace models)
        is_tuple = isinstance(attn_output, tuple)
        out_tensor = attn_output[0] if is_tuple else attn_output
        
        # ── Hebbian Fast-Weight update (zero-backprop, in-place) ─────────────
        with torch.no_grad():
            # Use the mean-pooled hidden state as the "pre-synaptic" signal
            pre = hidden_states.mean(dim=1)   # (batch, d_model)
            post = out_tensor.mean(dim=1)      # (batch, d_model)
            
            # Oja's rule: F ← (1 - η) * F + η * mean_batch(post ⊗ pre)
            outer = torch.einsum('bi,bj->ij', post, pre) / pre.shape[0]
            self.fast_weight = (1 - self.eta) * self.fast_weight + self.eta * outer
            
            # Compute and add the Hebbian delta to the output
            # (batch, seq, d_model) = (batch, seq, d_model) @ (d_model, d_model)^T
            delta = (out_tensor @ self.fast_weight.T) * self.scale
            
        self.update_count += 1
        enhanced_out = out_tensor + delta
        
        if is_tuple:
            return (enhanced_out,) + attn_output[1:]
        return enhanced_out

    @property
    def saturation(self) -> float:
        """How 'charged' the fast weight matrix is (0 = empty, 1 = saturated)."""
        return min(1.0, self.fast_weight.abs().mean().item() * 10)

    def reset(self):
        """Clears the Fast-Weight matrix (useful for new conversation context)."""
        self.fast_weight.zero_()
        self.update_count = 0


class FZAAttentionPatcher:
    """
    Applies AttentionHebbianPatch to selected layers of any HuggingFace model.
    
    Usage:
        patcher = FZAAttentionPatcher(model, patch_layers=[0, 1, 2])
        patcher.apply()
        # model now has Hebbian fast-weights in its first 3 attention blocks
        print(patcher.stats())
        patcher.reset_all()
    """
    
    def __init__(self, model: nn.Module, patch_layers: Optional[list] = None, eta: float = 0.01, scale: float = 0.05):
        """
        Args:
            model:         The HuggingFace CausalLM model
            patch_layers:  Layer indices to patch (default: first 3 layers only to save compute)
            eta:           Hebbian learning rate
            scale:         Output injection scale
        """
        self.model = model
        self.patch_layers = patch_layers or [0, 1, 2]
        self.eta = eta
        self.scale = scale
        self.patches: list[AttentionHebbianPatch] = []
        self._applied = False
    
    def apply(self) -> int:
        """
        Finds the transformer layers and patches their self-attention modules.
        Works with Mistral, Llama, Gemma, and any model with model.model.layers.
        
        Returns the number of layers successfully patched.
        """
        if self._applied:
            print("⚠️ [AttentionPatch] 이미 적용됨 — 중복 적용 방지.")
            return len(self.patches)
        
        # Standard HuggingFace structure: model.model.layers[i].self_attn
        try:
            layers = self.model.model.layers
        except AttributeError:
            print("⚠️ [AttentionPatch] 모델 구조가 표준 형식이 아닙니다 (model.model.layers 없음).")
            return 0
        
        d_model = self.model.config.hidden_size
        patched = 0
        
        for i in self.patch_layers:
            if i >= len(layers):
                break
            
            original_attn = layers[i].self_attn
            patch = AttentionHebbianPatch(original_attn, d_model, eta=self.eta, scale=self.scale)
            layers[i].self_attn = patch
            self.patches.append(patch)
            patched += 1
        
        self._applied = True
        print(f"🧬 [AttentionPatch] {patched}개 어텐션 블록에 Hebbian Fast-Weight 주입 완료 (η={self.eta}, scale={self.scale})")
        return patched
    
    def stats(self) -> dict:
        """Returns live statistics about all patched layers."""
        return {
            f"layer_{i}": {
                "updates": p.update_count,
                "saturation": f"{p.saturation:.1%}",
            }
            for i, p in enumerate(self.patches)
        }
    
    def reset_all(self):
        """Resets all Fast-Weight matrices (useful at start of new conversation)."""
        for p in self.patches:
            p.reset()
        print(f"🔄 [AttentionPatch] {len(self.patches)}개 Fast-Weight 행렬 초기화 완료")
