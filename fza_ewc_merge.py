"""
fza_ewc_merge.py — EWC Cross-Pollination
==========================================
Merges Elastic Weight Consolidation (EWC) Fisher Information Matrix diagonals
from two or more FZA nodes to share Leaf-zone knowledge without touching
each other's personal Root Zones.

Biological metaphor:
  Trees share nutrients through the Mycorrhizal network (broker).
  But each tree maintains its OWN root system (Root Zone / personal facts).
  The Fisher diagonal is the "nutrient gradient" — it tells the merge
  function exactly which parameters are personal (high Fisher diagonal value
  in Root zone = "don't touch") vs shareable (low diagonal = open to change).

Usage:
  from fza_ewc_merge import merge_fisher_diagonals
  merged = merge_fisher_diagonals(self_fisher, peer_fisher, alpha=0.4, root_mask=root_params)
"""

import torch
from typing import Dict, Optional, List


def merge_fisher_diagonals(
    self_fisher: Dict[str, torch.Tensor],
    peer_fisher: Dict[str, torch.Tensor],
    alpha: float = 0.4,
    root_param_names: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Merges Fisher Information Matrix diagonals from self and peer.

    Args:
        self_fisher: Dict[param_name → Fisher diagonal tensor] for this node
        peer_fisher:  Dict[param_name → Fisher diagonal tensor] for the peer node
        alpha: float in [0, 1]. Fraction of peer knowledge to absorb.
               0.0 = ignore peer. 1.0 = fully adopt peer.
               Recommended: 0.2 - 0.5 for conservative shared learning.
        root_param_names: list of parameter names in the Root Zone (personal facts).
                          These will NOT be modified regardless of alpha.

    Returns:
        merged_fisher: Dict[param_name → merged Fisher diagonal tensor]
        
    Formula:
        merged[param] = (1 - alpha) * self_fisher[param] + alpha * peer_fisher[param]
        if param in root_param_names:
            merged[param] = self_fisher[param]  # Root zone is immutable
    """
    root_set = set(root_param_names or [])
    merged = {}
    
    all_params = set(self_fisher.keys()) | set(peer_fisher.keys())
    skipped_root = 0
    merged_count = 0
    
    for name in all_params:
        if name not in self_fisher:
            # Peer has knowledge we don't have yet — gentle absorption
            if name not in root_set:
                merged[name] = alpha * peer_fisher[name].clone()
            continue
        
        if name not in peer_fisher:
            merged[name] = self_fisher[name].clone()
            continue
        
        # Both have this parameter
        if name in root_set:
            # Root zone: personal memory is sacred — do not merge
            merged[name] = self_fisher[name].clone()
            skipped_root += 1
        else:
            # Leaf zone: absorb a fraction of peer knowledge
            merged[name] = (1.0 - alpha) * self_fisher[name] + alpha * peer_fisher[name]
            merged_count += 1
    
    print(f"🌱 [EWCMerge] 병합 완료: {merged_count}개 파라미터 통합, {skipped_root}개 루트 보호 (α={alpha})")
    return merged


def export_fisher(ewc_instance) -> Dict[str, torch.Tensor]:
    """
    Extracts the Fisher diagonal from an FZAEwc instance for transmission.
    All tensors are moved to CPU and detached for serialization.
    """
    if not hasattr(ewc_instance, '_fisher'):
        return {}
    return {
        name: tensor.detach().cpu()
        for name, tensor in ewc_instance._fisher.items()
    }


def import_fisher(ewc_instance, fisher_dict: Dict[str, torch.Tensor]):
    """
    Loads a merged Fisher diagonal back into an FZAEwc instance.
    """
    if not hasattr(ewc_instance, '_fisher'):
        ewc_instance._fisher = {}
    
    device = next(iter(ewc_instance._fisher.values())).device if ewc_instance._fisher else torch.device('cpu')
    
    for name, tensor in fisher_dict.items():
        ewc_instance._fisher[name] = tensor.to(device)
    
    print(f"📊 [EWCMerge] Fisher 대각 로드 완료: {len(fisher_dict)}개 파라미터")


def serialize_fisher(fisher_dict: Dict[str, torch.Tensor]) -> Dict[str, list]:
    """Converts Fisher tensors to JSON-serializable lists."""
    return {name: tensor.tolist() for name, tensor in fisher_dict.items()}


def deserialize_fisher(fisher_data: Dict[str, list]) -> Dict[str, torch.Tensor]:
    """Reconstructs Fisher tensors from JSON-deserialized lists."""
    return {name: torch.tensor(data) for name, data in fisher_data.items()}
