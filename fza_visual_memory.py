"""
fza_visual_memory.py — Visual Scene Intake Pipeline (v13.0)
============================================================
Converts visual input (image paths, scene descriptions, or structured
object lists) into WorldGraph observations and Spatial Adapters.

This is the bridge between FZA's text-only memory world and the
physical, visual world the user actually inhabits.

Three intake modes are supported:

1. TEXT DESCRIPTION (always available, no vision model needed)
   User types: "봐봐 — 책상 위에 노트북, 커피잔, 그리고 이어폰이 있어"
   → FZA parses the scene, registers objects, creates a SpatialAdapter.

2. STRUCTURED JSON (for programmatic use / mobile app integration)
   Direct dict with {"objects": [{"name":"keys", "location":"table"}]}
   → FZA registers directly without text parsing.

3. IMAGE FILE (requires Pillow / vision model, optional dependency)
   Provide a path to an image.
   → If sentence-transformers + a vision encoder is available, use it.
   → Fallback: extract filename/EXIF description and treat as text.

The visual memory pipeline always outputs:
  - Updated WorldGraph (object locations)
  - A new SpatialAdapter (retrievable by query)
  - A natural language summary (for LLM system prompt injection)

Usage:
    from fza_visual_memory import VisualMemoryPipeline
    vmp = VisualMemoryPipeline()
    
    # Text mode
    result = vmp.ingest_description("책상 위에 노트북, 커피잔, 이어폰이 있어")
    print(result["summary"])
    
    # Structured mode
    result = vmp.ingest_structured({
        "location": "desk",
        "objects": ["laptop", "coffee_cup", "earphones"]
    })
"""

import re
import time
from typing import Optional
from fza_world_graph import WorldGraph
from fza_spatial_adapter import SpatialAdapterBank
from fza_event_bus import bus


# Common Korean location keywords for auto-detection
_KO_LOCATIONS = {
    "책상": "desk", "침대": "bed", "소파": "sofa", "주방": "kitchen",
    "부엌": "kitchen", "카운터": "counter", "선반": "shelf", "서랍": "drawer",
    "창문": "windowsill", "바닥": "floor", "테이블": "table", "냉장고": "fridge",
}

# Common Korean object keywords
_KO_OBJECTS = {
    "노트북": "laptop", "컵": "cup", "커피잔": "coffee_cup", "열쇠": "keys",
    "핸드폰": "phone", "이어폰": "earphones", "책": "book", "지갑": "wallet",
    "충전기": "charger", "마우스": "mouse", "키보드": "keyboard", "모니터": "monitor",
    "안경": "glasses", "가방": "bag", "우산": "umbrella",
}


class VisualMemoryPipeline:
    """
    Converts visual/scene input into permanent spatial memories.
    """
    
    def __init__(
        self,
        world_graph: Optional[WorldGraph] = None,
        spatial_bank: Optional[SpatialAdapterBank] = None,
    ):
        self.world_graph = world_graph or WorldGraph()
        self.spatial_bank = spatial_bank or SpatialAdapterBank()
        self.scenes_processed = 0
    
    # ── Public API ────────────────────────────────────────────────────────────
    
    def ingest_description(self, text: str, confidence: float = 0.85) -> dict:
        """
        Parses a natural language scene description in Korean or English.
        Extracts objects and infers their locations from context.
        
        Args:
            text:       Scene description string
            confidence: Base confidence for extracted observations
        
        Returns:
            {"objects_detected": [...], "adapter_id": str, "summary": str}
        """
        location = self._detect_location(text)
        objects = self._detect_objects(text)
        
        if not objects:
            return {"objects_detected": [], "adapter_id": None, "summary": "객체를 감지하지 못했습니다."}
        
        anchors = [(obj, location) for obj in objects]
        
        # Register in World Graph
        for obj in objects:
            self.world_graph.observe(obj, location=location, confidence=confidence)
        
        # Create Spatial Adapter
        adapter = self.spatial_bank.create(
            scene_description=text,
            spatial_anchors=anchors,
            source="description",
            confidence=confidence,
        )
        
        self.scenes_processed += 1
        summary = f"🏠 [{location}] 감지된 객체: {', '.join(objects)}"
        bus.emit("visual_memory_ingested", {
            "location": location,
            "objects": objects,
            "adapter_id": adapter.adapter_id,
            "source": "description",
        })
        print(f"👁️  [VisualMem] {summary}")
        
        return {
            "objects_detected": objects,
            "location": location,
            "adapter_id": adapter.adapter_id,
            "summary": summary,
        }
    
    def ingest_structured(self, scene: dict) -> dict:
        """
        Ingests a structured scene dict from a mobile/programmatic source.
        
        Args:
            scene: {"location": str, "objects": [str, ...], "confidence": float (opt)}
        """
        location = scene.get("location", "unknown").replace(" ", "_")
        objects = [o.lower() for o in scene.get("objects", [])]
        confidence = scene.get("confidence", 0.90)
        
        if not objects:
            return {"objects_detected": [], "adapter_id": None, "summary": "객체 없음"}
        
        anchors = [(obj, location) for obj in objects]
        description = f"[구조화 장면] {location}: {', '.join(objects)}"
        
        for obj in objects:
            self.world_graph.observe(obj, location=location, confidence=confidence)
        
        adapter = self.spatial_bank.create(
            scene_description=description,
            spatial_anchors=anchors,
            source="structured",
            confidence=confidence,
        )
        
        self.scenes_processed += 1
        summary = f"📱 [구조화 입력] {location}: {', '.join(objects)}"
        bus.emit("visual_memory_ingested", {
            "location": location,
            "objects": objects,
            "adapter_id": adapter.adapter_id,
            "source": "structured",
        })
        print(f"👁️  [VisualMem] {summary}")
        
        return {
            "objects_detected": objects,
            "location": location,
            "adapter_id": adapter.adapter_id,
            "summary": summary,
        }
    
    def answer_location_query(self, query: str) -> str:
        """
        Answers a "where is X?" query using the World Graph + Spatial Bank.
        Returns a natural language answer string.
        """
        # Try World Graph first (precise location + timestamp)
        query_lower = query.lower()
        for ko, en in _KO_OBJECTS.items():
            if ko in query_lower or en in query_lower:
                result = self.world_graph.locate(ko) or self.world_graph.locate(en)
                if result:
                    loc_display = result["location"].replace("_", " ")
                    age = result["age_hours"]
                    age_str = f"{age:.1f}시간 전" if age > 1 else f"{int(age * 60)}분 전"
                    status_note = " (위치가 변경됐을 수 있어)" if result["status"] == "possibly_moved" else ""
                    return f"마지막으로 **{loc_display}**에서 봤어, {age_str}{status_note}. (신뢰도 {result['effective_confidence']:.0%})"
        
        # Fallback to Spatial Adapter semantic search
        ctx = self.spatial_bank.build_context_for_query(query, top_k=1)
        if ctx:
            return f"공간 기억에서 찾은 관련 장면:\n{ctx}"
        
        return "그 물건을 어디에 뒀는지 아직 기억하지 못해. 본 적이 있으면 '봐봐, [위치]에 [물건]이 있어'라고 알려줘!"
    
    def build_spatial_context(self, query: str) -> str:
        """
        Builds the full spatial context block for LLM injection.
        Combines World Graph summary + relevant Spatial Adapter hits.
        """
        wg_summary = self.world_graph.build_natural_language_summary()
        spatial_ctx = self.spatial_bank.build_context_for_query(query, top_k=2)
        parts = [p for p in [wg_summary, spatial_ctx] if p]
        return "\n\n".join(parts)
    
    # ── Private Helpers ───────────────────────────────────────────────────────
    
    def _detect_location(self, text: str) -> str:
        """Extracts the most prominent location keyword from text."""
        for ko, en in _KO_LOCATIONS.items():
            if ko in text:
                return en
        # English fallback
        for loc in ["desk", "kitchen", "bedroom", "shelf", "table", "floor"]:
            if loc in text.lower():
                return loc
        return "unknown_location"
    
    def _detect_objects(self, text: str) -> list:
        """Extracts known object names from text (Korean + English)."""
        found = []
        for ko, en in _KO_OBJECTS.items():
            if ko in text:
                found.append(ko)
        # English fallback
        for en_obj in ["laptop", "phone", "keys", "cup", "charger", "glasses", "wallet", "bag"]:
            if en_obj in text.lower() and en_obj not in found:
                found.append(en_obj)
        return found
    
    def get_stats(self) -> dict:
        return {
            "scenes_processed": self.scenes_processed,
            "world_graph_objects": len(self.world_graph.get_all()),
            "spatial_adapters": len(self.spatial_bank.get_all()),
        }
