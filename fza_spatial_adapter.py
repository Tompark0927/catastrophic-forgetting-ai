"""
fza_spatial_adapter.py — Spatial Context LoRA Adapters (v13.0)
===============================================================
Extends the standard LoRA adapter bank to encode spatial / scene context
alongside the usual semantic content.

A Spatial Adapter wraps a regular FZA memory adapter and adds:
  - scene_description: natural language description of the visual scene
  - spatial_anchors: list of (object, location) pairs from that scene
  - timestamp: when this scene was observed
  - source: "camera", "description", "recalled"

When the Memory Graph routes a query, spatial adapters compete alongside
text-only adapters. A query like "내 열쇠 어디 있어?" will have high
cosine similarity to spatial adapters that contain "열쇠" observations.

Biological parallel: The Hippocampus binds together "what" (object) with
"where" (location) into an episodic memory — the same binding is done here
by combining a semantic text adapter with a spatial anchor list.

Usage:
    from fza_spatial_adapter import SpatialAdapter, SpatialAdapterBank
    bank = SpatialAdapterBank("./adapters/spatial")
    
    adapter = bank.create(
        scene_description="사무실 책상 위에 노트북과 커피잔이 있다",
        spatial_anchors=[("laptop", "desk"), ("coffee_cup", "desk")],
        source="description",
    )
    
    results = bank.search("노트북 어디 있어?")
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict


@dataclass
class SpatialAdapter:
    """
    An adapter that encodes a spatial scene in natural language
    alongside structured (object, location) anchor pairs.
    """
    adapter_id: str
    scene_description: str
    spatial_anchors: List[Tuple[str, str]]   # [(object_name, location), ...]
    timestamp: float = field(default_factory=time.time)
    source: str = "description"              # "camera" | "description" | "recalled"
    confidence: float = 1.0
    
    def to_prompt_text(self) -> str:
        """Converts this adapter to text suitable for LLM context injection."""
        anchors_str = ", ".join(f"{obj}({loc})" for obj, loc in self.spatial_anchors)
        age_h = (time.time() - self.timestamp) / 3600
        age_str = f"{age_h:.1f}시간 전" if age_h > 1 else f"{int(age_h * 60)}분 전"
        return f"[공간 장면 {age_str}] {self.scene_description} | 위치: {anchors_str}"
    
    def matches_query(self, query: str) -> float:
        """
        Simple keyword-overlap scoring between query and this adapter's content.
        Returns a score in [0, 1].
        (In production: replace with sentence-transformer cosine similarity.)
        """
        q_lower = query.lower()
        target = (self.scene_description.lower() + " " +
                  " ".join(f"{o} {l}" for o, l in self.spatial_anchors))
        
        q_words = set(q_lower.split())
        t_words = set(target.split())
        overlap = len(q_words & t_words)
        score = overlap / max(1, len(q_words))
        
        # Penalize stale observations
        age_h = (time.time() - self.timestamp) / 3600
        staleness_penalty = min(0.5, age_h / 48.0)
        return max(0.0, score - staleness_penalty) * self.confidence
    
    def to_dict(self) -> dict:
        return asdict(self)


class SpatialAdapterBank:
    """
    A persistent store for Spatial Adapters.
    Stored as individual JSON files under `base_dir/spatial/`.
    """
    
    def __init__(self, base_dir: str = "./adapters/spatial"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._cache: Dict[str, SpatialAdapter] = {}
        self._load_all()
    
    def create(
        self,
        scene_description: str,
        spatial_anchors: List[Tuple[str, str]],
        source: str = "description",
        confidence: float = 1.0,
    ) -> SpatialAdapter:
        """Creates and persists a new Spatial Adapter."""
        adapter = SpatialAdapter(
            adapter_id=str(uuid.uuid4())[:12],
            scene_description=scene_description,
            spatial_anchors=spatial_anchors,
            source=source,
            confidence=confidence,
        )
        self._cache[adapter.adapter_id] = adapter
        self._save(adapter)
        print(f"🗺️  [SpatialAdapter] 새 공간 기억 생성: '{scene_description[:50]}' | {len(spatial_anchors)}개 앵커")
        return adapter
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[SpatialAdapter, float]]:
        """
        Searches all spatial adapters for those most relevant to the query.
        Returns a list of (adapter, score) tuples sorted by score descending.
        """
        if not self._cache:
            return []
        
        scored = []
        for adapter in self._cache.values():
            score = adapter.matches_query(query)
            if score > 0.0:
                scored.append((adapter, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def build_context_for_query(self, query: str, top_k: int = 2) -> str:
        """
        Returns a formatted text block of the most relevant spatial memories
        for injection into the LLM prompt.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        
        lines = ["[관련 공간 기억]"]
        for adapter, score in results:
            lines.append(f"  • {adapter.to_prompt_text()} (관련도 {score:.0%})")
        return "\n".join(lines)
    
    def get_all(self) -> List[SpatialAdapter]:
        return list(self._cache.values())
    
    def _save(self, adapter: SpatialAdapter):
        path = os.path.join(self.base_dir, f"{adapter.adapter_id}.json")
        with open(path, "w") as f:
            json.dump(adapter.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _load_all(self):
        loaded = 0
        for fname in os.listdir(self.base_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self.base_dir, fname)) as f:
                        data = json.load(f)
                    data["spatial_anchors"] = [tuple(a) for a in data["spatial_anchors"]]
                    adapter = SpatialAdapter(**data)
                    self._cache[adapter.adapter_id] = adapter
                    loaded += 1
                except Exception as e:
                    print(f"⚠️ [SpatialAdapter] 로드 실패 ({fname}): {e}")
        if loaded:
            print(f"🗺️  [SpatialAdapter] {loaded}개 공간 어댑터 로드 완료")
