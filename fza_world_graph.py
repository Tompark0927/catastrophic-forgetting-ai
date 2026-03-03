"""
fza_world_graph.py — 3D Spatial Object Graph (v13.0)
=====================================================
FZA's spatial memory system. Remembers WHERE things are in the
user's physical world — not just WHAT they are.

Biological metaphor: The Hippocampus's "Place Cells" and "Grid Cells".
In mammals, dedicated neurons fire specifically when the animal is at a
particular location. This module is that system for FZA: each object
the AI encounters is encoded with spatial coordinates (room, position)
and time-stamps, giving the AI a living mental map of the user's world.

Examples of what this enables:
  - "Where are my keys?" → checks world graph for last known location
  - "Is the coffee maker still in the kitchen?" → recalls scene + time
  - "I moved my monitor from the desk to the shelf" → updates graph
  - Object permanence: items not seen recently are flagged as "possibly moved"

Usage:
    from fza_world_graph import WorldGraph
    wg = WorldGraph()
    wg.observe("keys", location="kitchen_counter", confidence=0.95)
    wg.observe("laptop", location="desk", confidence=0.90)
    
    result = wg.locate("keys")
    print(result)  # → {"location": "kitchen_counter", "last_seen": <timestamp>}
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class SpatialObject:
    """An object with a known location and confidence score."""
    name: str
    location: str                       # e.g. "kitchen_counter", "desk", "living_room"
    confidence: float = 1.0             # 0.0 = uncertain, 1.0 = certain
    last_seen: float = field(default_factory=time.time)
    observation_count: int = 1
    notes: str = ""
    
    @property
    def age_hours(self) -> float:
        return (time.time() - self.last_seen) / 3600
    
    @property
    def staleness(self) -> float:
        """How stale is this observation? 0.0=fresh, 1.0=very old (>24h)"""
        return min(1.0, self.age_hours / 24.0)
    
    @property
    def effective_confidence(self) -> float:
        """Confidence adjusted for staleness."""
        return self.confidence * (1.0 - self.staleness * 0.5)
    
    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "age_hours": round(self.age_hours, 2),
            "effective_confidence": round(self.effective_confidence, 3),
        }


class WorldGraph:
    """
    An associative spatial memory store. Maps object names to their
    last known locations, confidence scores, and timestamps.
    
    Persisted to disk as JSON at `world_graph.json`.
    """
    
    SAVE_PATH = "world_graph.json"
    STALE_THRESHOLD = 0.5   # Below this effective_confidence → "possibly moved"
    
    def __init__(self, save_path: str = SAVE_PATH):
        self.save_path = save_path
        self._objects: Dict[str, SpatialObject] = {}
        self._location_index: Dict[str, List[str]] = {}  # location → [object_names]
        self._load()
    
    def observe(self, name: str, location: str, confidence: float = 1.0, notes: str = "") -> SpatialObject:
        """
        Records an observation of an object at a specific location.
        If the object is already known, updates its location.
        
        Args:
            name:       The object name (lowercase, e.g. "keys", "laptop")
            location:   The spatial location (e.g. "kitchen_counter", "desk")
            confidence: Observation confidence 0.0-1.0
            notes:      Optional contextual note
        """
        name = name.lower().strip()
        location = location.lower().strip().replace(" ", "_")
        
        if name in self._objects:
            old = self._objects[name]
            moved = old.location != location
            old.location = location
            old.confidence = confidence
            old.last_seen = time.time()
            old.observation_count += 1
            if notes:
                old.notes = notes
            if moved:
                print(f"🔄 [WorldGraph] '{name}' 이동 감지: {old.location} → {location}")
            obj = old
        else:
            obj = SpatialObject(name=name, location=location, confidence=confidence, notes=notes)
            self._objects[name] = obj
            print(f"📍 [WorldGraph] 새 객체 등록: '{name}' @ {location} (신뢰도 {confidence:.0%})")
        
        # Update location index
        for loc, names in self._location_index.items():
            if name in names:
                names.remove(name)
        self._location_index.setdefault(location, [])
        if name not in self._location_index[location]:
            self._location_index[location].append(name)
        
        self._save()
        return obj
    
    def locate(self, name: str) -> Optional[dict]:
        """
        Returns the last known location of an object.
        Includes a staleness warning if the observation is old.
        """
        name = name.lower().strip()
        obj = self._objects.get(name)
        if not obj:
            return None
        
        result = obj.to_dict()
        result["status"] = "possibly_moved" if obj.effective_confidence < self.STALE_THRESHOLD else "likely_here"
        return result
    
    def scan_location(self, location: str) -> List[dict]:
        """Returns all objects last seen at a specific location."""
        location = location.lower().strip().replace(" ", "_")
        names = self._location_index.get(location, [])
        return [self._objects[n].to_dict() for n in names if n in self._objects]
    
    def build_natural_language_summary(self) -> str:
        """
        Generates a natural language summary of the world graph for injection
        into the LLM's system prompt.
        
        Example output:
            [공간 기억] 마지막으로 알려진 위치:
            - 열쇠: 주방 카운터 (3.2시간 전, 신뢰도 87%)
            - 노트북: 책상 (0.5시간 전, 신뢰도 95%)
        """
        if not self._objects:
            return ""
        
        lines = ["[공간 기억] 마지막으로 알려진 위치:"]
        for name, obj in sorted(self._objects.items(), key=lambda x: x[1].last_seen, reverse=True):
            loc_display = obj.location.replace("_", " ")
            age_display = f"{obj.age_hours:.1f}시간 전" if obj.age_hours > 1 else f"{int(obj.age_hours * 60)}분 전"
            conf_display = f"{obj.effective_confidence:.0%}"
            status = " ⚠️(위치 불확실)" if obj.effective_confidence < self.STALE_THRESHOLD else ""
            lines.append(f"- {name}: {loc_display} ({age_display}, 신뢰도 {conf_display}){status}")
        
        return "\n".join(lines)
    
    def get_all(self) -> List[dict]:
        return [o.to_dict() for o in self._objects.values()]
    
    def _save(self):
        data = {name: asdict(obj) for name, obj in self._objects.items()}
        try:
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ [WorldGraph] 저장 실패: {e}")
    
    def _load(self):
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path) as f:
                data = json.load(f)
            for name, d in data.items():
                self._objects[name] = SpatialObject(**{k: v for k, v in d.items() if k in SpatialObject.__dataclass_fields__})
                loc = self._objects[name].location
                self._location_index.setdefault(loc, [])
                if name not in self._location_index[loc]:
                    self._location_index[loc].append(name)
            print(f"🗺️  [WorldGraph] {len(self._objects)}개 공간 객체 로드 완료")
        except Exception as e:
            print(f"⚠️ [WorldGraph] 로드 실패: {e}")
