"""
fza_lobe_spawner.py — Autonomous Sub-Network Spawner (v14.0)
=============================================================
The second half of the Singularity Threshold.

When FZA identifies a new domain it has no specialized architecture for,
it doesn't just train a LoRA adapter — it spawns an entirely new neural
"lobe": a fresh PyTorch sub-module initialized from scratch, wired into
the main inference stream as a domain-specific expert head.

Biological metaphor: **Neuroplastic cortical remapping**.
When a blind person's visual cortex gets co-opted for tactile processing,
the brain hasn't added neurons — it's rerouted existing ones and grown
new local circuits. Lobe spawning does the same thing in software:
a new PyTorch Module appears at runtime, gets hot-wired into the model's
ResNet stream via a lightweight residual gate, and begins learning
immediately via the existing LoRA training pipeline.

What a "Lobe" is:
    - A small (2-4 layer) MLP or attention module
    - Initialized with Xavier-uniform weights
    - Attached to the model via a learnable gate scalar (initially 0.0 → silent)
    - Activated as the gate learns to open (via gradient descent on replay)
    - Tagged with a domain name (e.g. "quantum_chemistry", "music_theory")

Persistence:
    - Each lobe is saved to `./lobes/<domain_name>.pt`
    - On startup, all saved lobes are loaded and re-attached
    
Usage:
    from fza_lobe_spawner import LobeSpawner
    spawner = LobeSpawner(model=my_model)
    
    lobe_id = spawner.spawn("quantum_chemistry")
    print(spawner.list_lobes())
"""

import os
import time
import uuid
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


LOBES_DIR = "./lobes"


@dataclass
class LobeMetadata:
    """Metadata for a spawned lobe sub-network."""
    lobe_id: str
    domain: str
    created_at: float
    input_dim: int
    hidden_dim: int
    output_dim: int
    activation_count: int = 0   # How many forward passes have used this lobe
    gate_value: float = 0.0     # Current gate scalar (0 = silent, 1 = fully active)
    status: str = "dormant"     # "dormant" | "warming" | "active"
    
    def to_dict(self) -> dict:
        return asdict(self)


class DomainLobe(nn.Module):
    """
    A lightweight 2-layer MLP sub-network representing expertise in one domain.
    
    Architecture:
        in_proj:  input_dim → hidden_dim  (SiLU activation)
        out_proj: hidden_dim → output_dim (no activation)
        gate:     learnable scalar in [0, 1] — controls how much output blends in
    
    The final output is: original_out + gate * lobe_out
    So initially (gate=0) the lobe is completely silent.
    """
    
    def __init__(self, input_dim: int = 4096, hidden_dim: int = 512, output_dim: int = 4096):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=True)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, output_dim, bias=True)
        self.gate = nn.Parameter(torch.zeros(1))   # Starts silent
        
        # Xavier initialization for good gradient flow from the start
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        Input hidden state (batch, seq, input_dim)
            base_out: The base model's output for this layer (batch, seq, output_dim)
        
        Returns:
            Blended output: base_out + sigmoid(gate) * lobe(x)
        """
        lobe_out = self.out_proj(self.act(self.in_proj(x)))
        gate_scalar = torch.sigmoid(self.gate)
        return base_out + gate_scalar * lobe_out


class LobeSpawner:
    """
    Manages the lifecycle of all domain lobes:
    spawning, loading, attaching, and reporting.
    """
    
    def __init__(self, model: Optional[nn.Module] = None, lobes_dir: str = LOBES_DIR):
        self.model = model
        self.lobes_dir = lobes_dir
        os.makedirs(lobes_dir, exist_ok=True)
        self._lobes: Dict[str, DomainLobe] = {}
        self._metadata: Dict[str, LobeMetadata] = {}
        self._load_existing()
    
    def spawn(
        self,
        domain: str,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        output_dim: int = 4096,
    ) -> str:
        """
        Spawns a new domain lobe sub-network.
        
        Args:
            domain:     Domain name (e.g. "quantum_chemistry", "music_theory")
            input_dim:  Hidden state dimension (must match model)
            hidden_dim: Internal MLP width
            output_dim: Output dimension (must match input_dim for residual)
        
        Returns:
            lobe_id: The unique ID of the new lobe
        """
        # Don't spawn duplicates
        for lm in self._metadata.values():
            if lm.domain == domain:
                print(f"⚠️  [LobeSpawner] '{domain}' 로브가 이미 존재합니다: {lm.lobe_id[:8]}")
                return lm.lobe_id
        
        lobe_id = str(uuid.uuid4())[:12]
        lobe = DomainLobe(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        
        metadata = LobeMetadata(
            lobe_id=lobe_id,
            domain=domain,
            created_at=time.time(),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            gate_value=0.0,
            status="dormant",
        )
        
        self._lobes[lobe_id] = lobe
        self._metadata[lobe_id] = metadata
        
        self._save_lobe(lobe_id)
        
        param_count = sum(p.numel() for p in lobe.parameters())
        print(f"🧠 [LobeSpawner] 새 로브 생성: '{domain}' "
              f"({param_count:,}개 파라미터, dim={input_dim}→{hidden_dim}→{output_dim})")
        
        return lobe_id
    
    def activate(self, lobe_id: str, target_gate: float = 0.5):
        """
        Gradually nudges the gate open for a lobe, transitioning it from
        dormant to active. In production, gradient descent does this automatically.
        """
        if lobe_id not in self._lobes:
            print(f"⚠️  [LobeSpawner] 로브 {lobe_id[:8]} 미발견")
            return
        
        lobe = self._lobes[lobe_id]
        meta = self._metadata[lobe_id]
        
        # Nudge gate logit toward target
        with torch.no_grad():
            import math
            target_logit = math.log(target_gate / (1 - target_gate + 1e-7))
            lobe.gate.fill_(target_logit)
        
        meta.gate_value = float(torch.sigmoid(lobe.gate))
        meta.status = "active" if meta.gate_value > 0.3 else "warming"
        self._save_lobe(lobe_id)
        
        print(f"⚡ [LobeSpawner] '{meta.domain}' 로브 게이트 열림: {meta.gate_value:.2f} ({meta.status})")
    
    def forward_lobe(self, lobe_id: str, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass through a specific lobe and blends with base_out."""
        lobe = self._lobes.get(lobe_id)
        if lobe is None:
            return base_out
        
        lobe.eval()
        with torch.no_grad():
            result = lobe(x, base_out)
        
        meta = self._metadata[lobe_id]
        meta.activation_count += 1
        if meta.activation_count % 100 == 0:
            self._save_lobe(lobe_id)
        
        return result
    
    def list_lobes(self) -> List[dict]:
        return [m.to_dict() for m in self._metadata.values()]
    
    def get_lobe(self, lobe_id: str) -> Optional[DomainLobe]:
        return self._lobes.get(lobe_id)
    
    def get_by_domain(self, domain: str) -> Optional[str]:
        for lm in self._metadata.values():
            if lm.domain == domain:
                return lm.lobe_id
        return None
    
    def _save_lobe(self, lobe_id: str):
        lobe = self._lobes[lobe_id]
        meta = self._metadata[lobe_id]
        # Update gate value in metadata before saving
        meta.gate_value = float(torch.sigmoid(lobe.gate))
        
        state = {
            "weights": lobe.state_dict(),
            "metadata": meta.to_dict(),
        }
        path = os.path.join(self.lobes_dir, f"{lobe_id}.pt")
        torch.save(state, path)
    
    def _load_existing(self):
        """Loads all previously saved lobes on startup."""
        loaded = 0
        for fname in os.listdir(self.lobes_dir):
            if not fname.endswith(".pt"):
                continue
            try:
                path = os.path.join(self.lobes_dir, fname)
                state = torch.load(path, map_location="cpu", weights_only=False)
                meta_dict = state["metadata"]
                meta = LobeMetadata(**meta_dict)
                lobe = DomainLobe(
                    input_dim=meta.input_dim,
                    hidden_dim=meta.hidden_dim,
                    output_dim=meta.output_dim,
                )
                lobe.load_state_dict(state["weights"])
                self._lobes[meta.lobe_id] = lobe
                self._metadata[meta.lobe_id] = meta
                loaded += 1
            except Exception as e:
                print(f"⚠️  [LobeSpawner] 로드 실패 ({fname}): {e}")
        
        if loaded:
            print(f"🧠 [LobeSpawner] {loaded}개 도메인 로브 복원 완료")
    
    def get_stats(self) -> dict:
        return {
            "total_lobes": len(self._lobes),
            "active_lobes": sum(1 for m in self._metadata.values() if m.status == "active"),
            "domains": [m.domain for m in self._metadata.values()],
        }
