"""
fza_broker.py — The Mycorrhizal Hub (port 8001)
================================================
A pure-coordination FastAPI server. No LLM is loaded here — this is
purely an adapter registry and knowledge-transfer relay.

Biological metaphor: this is the mycorrhizal fungal network itself.
Thin fiber threads (HTTP) carry encoded adapter "spores" between
FZA nodes (tree root systems). The broker routes them but never
modifies them.

Usage:
  source .venv/bin/activate
  python fza_broker.py

Endpoints:
  POST /adapters/upload    — receive a packed adapter blob
  GET  /adapters/list      — list all available adapters
  GET  /adapters/{id}      — download a specific adapter
  DELETE /adapters/{id}    — remove an adapter
  WS   /ws                 — real-time event stream for connected nodes
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from fza_node_registry import NodeRegistry

# In-memory store: adapter_id → {blob, metadata, timestamp}
_adapter_store: Dict[str, Any] = {}
_ws_connections: Dict[str, WebSocket] = {}
_main_loop = None
_node_registry = NodeRegistry()   # v11.0 — live Hive-Mind node catalog


def _broadcast_sync(event: dict):
    """Thread-safe broadcast to all connected WebSocket clients."""
    if _main_loop and not _main_loop.is_closed():
        _main_loop.call_soon_threadsafe(
            lambda: _main_loop.create_task(_broadcast_async(event))
        )


async def _broadcast_async(event: dict):
    for ws in list(_ws_connections.values()):
        try:
            await ws.send_json(event)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    print("🍄 [Broker] Mycorrhizal Hub online — port 8001")
    yield
    print("🛑 [Broker] Shutting down.")


app = FastAPI(lifespan=lifespan, title="FZA Mycorrhizal Broker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cid = str(uuid.uuid4())
    _ws_connections[cid] = ws
    await ws.send_json({"type": "node_connected", "node_id": cid, "adapter_count": len(_adapter_store)})
    print(f"🔗 [Broker] Node 연결됨: {cid[:8]}")
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(30)
            await ws.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        del _ws_connections[cid]
        print(f"🔌 [Broker] Node 연결 해제: {cid[:8]}")


@app.post("/adapters/upload")
async def upload_adapter(payload: dict):
    """
    Accepts a packed adapter blob from any FZA node.
    The payload must follow the format produced by fza_sync_protocol.pack_adapter().
    """
    adapter_id = payload.get("adapter_id")
    if not adapter_id:
        raise HTTPException(status_code=400, detail="adapter_id is required")
    
    _adapter_store[adapter_id] = {
        "blob": payload,
        "timestamp": time.time(),
        "metadata": payload.get("metadata", {}),
    }
    
    event = {
        "type": "adapter_uploaded",
        "adapter_id": adapter_id,
        "metadata": payload.get("metadata", {}),
        "file_count": len(payload.get("files", [])),
    }
    await _broadcast_async(event)
    print(f"📥 [Broker] 어댑터 수신: {adapter_id[:8]} ({len(payload.get('files', []))}개 파일)")
    
    return {"status": "ok", "adapter_id": adapter_id}


@app.get("/adapters/list")
async def list_adapters():
    """Returns the adapter catalog."""
    catalog = []
    for aid, entry in _adapter_store.items():
        catalog.append({
            "adapter_id": aid,
            "timestamp": entry["timestamp"],
            "metadata": entry["metadata"],
        })
    return {"adapters": catalog, "total": len(catalog)}


@app.get("/adapters/{adapter_id}")
async def get_adapter(adapter_id: str):
    """Downloads a specific adapter blob."""
    if adapter_id not in _adapter_store:
        raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")
    
    event = {"type": "adapter_downloaded", "adapter_id": adapter_id}
    await _broadcast_async(event)
    print(f"📤 [Broker] 어댑터 전송: {adapter_id[:8]}")
    
    return _adapter_store[adapter_id]["blob"]


@app.delete("/adapters/{adapter_id}")
async def delete_adapter(adapter_id: str):
    """Removes an adapter from the registry."""
    if adapter_id not in _adapter_store:
        raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")
    del _adapter_store[adapter_id]
    print(f"🗑️ [Broker] 어댑터 삭제: {adapter_id[:8]}")
    return {"status": "deleted"}


@app.get("/status")
async def status():
    alive = _node_registry.get_alive()
    return {
        "status": "online",
        "adapters": len(_adapter_store),
        "ws_connections": len(_ws_connections),
        "hive_nodes": len(alive),
    }


# ── v11.0: Node Registry Endpoints ──────────────────────────────────────────
@app.post("/nodes/register")
async def register_node(payload: dict):
    """
    A FZA node calls this endpoint when it boots to announce itself.
    Payload: { host, port, adapter_count, adapter_topics, device, node_id (optional) }
    """
    node = _node_registry.register(
        host=payload.get("host", "localhost"),
        port=payload.get("port", 8000),
        adapter_count=payload.get("adapter_count", 0),
        adapter_topics=payload.get("adapter_topics", []),
        device=payload.get("device", "cpu"),
        node_id=payload.get("node_id"),
    )
    await _broadcast_async({"type": "node_registered", "node_id": node.node_id, "host": node.host})
    return {"status": "registered", "node_id": node.node_id}


@app.post("/nodes/heartbeat")
async def node_heartbeat(payload: dict):
    """FZA nodes call this every 30s to stay marked as alive."""
    node_id = payload.get("node_id")
    if not node_id:
        raise HTTPException(status_code=400, detail="node_id required")
    ok = _node_registry.heartbeat(
        node_id=node_id,
        adapter_count=payload.get("adapter_count"),
        load=payload.get("load"),
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Node not registered")
    return {"status": "ok"}


@app.get("/nodes/list")
async def list_nodes():
    """Returns all currently alive FZA nodes and their metadata."""
    return _node_registry.summary()


@app.post("/nodes/find_experts")
async def find_experts(payload: dict):
    """
    Given a list of query topics, returns the top-K most relevant expert nodes.
    Payload: { topics: [str], top_k: int }
    """
    topics = payload.get("topics", [])
    top_k = payload.get("top_k", 2)
    experts = _node_registry.find_experts(topics, top_k=top_k)
    return {"experts": [e.to_dict() for e in experts], "queried_topics": topics}


@app.delete("/nodes/{node_id}")
async def unregister_node(node_id: str):
    """Manually removes a node from the registry."""
    _node_registry.unregister(node_id)
    return {"status": "unregistered"}


if __name__ == "__main__":
    uvicorn.run("fza_broker:app", host="0.0.0.0", port=8001, reload=False)
