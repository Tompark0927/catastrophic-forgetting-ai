"""
fza_sync_protocol.py — The Fungal Thread
=========================================
Serializes and deserializes FZA LoRA adapter delta weights into a compact JSON
blob suitable for transmission across the Mycorrhizal Network (broker, HTTP, etc).

The adapters themselves are tiny (~MB, not GB) — they are only the LoRA delta
matrices, not the full 7B base model. This is the biological equivalent of fungal
spores: lightweight, self-contained packets of encoded knowledge.
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Optional


def pack_adapter(adapter_id: str, adapter_path: str, metadata: Optional[dict] = None) -> dict:
    """
    Packs a LoRA adapter directory into a JSON-serializable blob for transport.
    
    The adapter directory (from peft.save_pretrained) typically contains:
    - adapter_model.safetensors (or adapter_model.bin) — the delta weights
    - adapter_config.json — the LoRA config
    
    Returns a dict with:
    - adapter_id
    - metadata (facts preview, timestamp)
    - files: [{name, content_b64}]
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    files = []
    for file_path in adapter_dir.iterdir():
        if file_path.is_file():
            with open(file_path, "rb") as f:
                content = f.read()
            files.append({
                "name": file_path.name,
                "content_b64": base64.b64encode(content).decode("utf-8"),
                "size_bytes": len(content),
            })
    
    blob = {
        "adapter_id": adapter_id,
        "timestamp": time.time(),
        "metadata": metadata or {},
        "files": files,
    }
    
    total_size = sum(f["size_bytes"] for f in files)
    print(f"📦 [SyncProtocol] 어댑터 패킹 완료: {adapter_id[:8]} ({total_size / 1024:.1f} KB, {len(files)}개 파일)")
    return blob


def unpack_adapter(blob: dict, destination_dir: str) -> str:
    """
    Unpacks a transport blob back into a local adapter directory.
    
    Returns the adapter_id.
    """
    adapter_id = blob["adapter_id"]
    adapter_dir = Path(destination_dir) / adapter_id
    adapter_dir.mkdir(parents=True, exist_ok=True)
    
    for file_info in blob.get("files", []):
        content = base64.b64decode(file_info["content_b64"])
        out_path = adapter_dir / file_info["name"]
        with open(out_path, "wb") as f:
            f.write(content)
    
    print(f"📬 [SyncProtocol] 어댑터 언패킹 완료: {adapter_id[:8]} → {adapter_dir}")
    return adapter_id


def serialize(blob: dict) -> str:
    """Serialize a packed blob to a JSON string."""
    return json.dumps(blob)


def deserialize(json_str: str) -> dict:
    """Deserialize a JSON string to a packed blob."""
    return json.loads(json_str)
