"""
fza_cloud_broker.py — Swarm Coordination Broker (v16.0)
========================================================
A lightweight FastAPI broker that acts as the swarm's central registry.

While `fza_broker.py` coordinates adapter/knowledge sharing between nodes,
`fza_cloud_broker.py` handles the *compute layer* — tracking which child
nodes exist, their load, and routing overflow requests to them.

Endpoints:
    POST /swarm/register      — A child node announces itself when it boots
    POST /swarm/heartbeat     — Periodic keepalive from each child node
    GET  /swarm/nodes         — List all registered live nodes
    POST /swarm/dispatch      — Route a query to the best available child
    DELETE /swarm/node/{id}   — Remove a node from the registry (culling)
    GET  /swarm/status        — Combined swarm health dashboard

This broker is intentionally minimal: it does NOT run any LLM itself.
It is pure traffic coordination, designed to run on a minimal CPU-only server.

Start:
    uvicorn fza_cloud_broker:app --host 0.0.0.0 --port 8002
"""

import time
import random
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI(
    title="FZA Cloud Broker",
    description="Swarm Coordination Broker for FZA v16.0 — Asexual Compute Replication",
    version="16.0.0",
)

# ── In-memory node registry ─────────────────────────────────────────────────
_swarm_nodes: Dict[str, dict] = {}
_HEARTBEAT_TIMEOUT = 90   # seconds — node is considered dead after this


# ── Pydantic models ─────────────────────────────────────────────────────────

class NodeRegistration(BaseModel):
    node_id: str
    label: str
    host: str
    port: int
    parent_id: Optional[str] = None

class NodeHeartbeat(BaseModel):
    node_id: str
    load: float = 0.0      # 0.0 = idle, 1.0 = fully loaded
    requests_served: int = 0

class DispatchRequest(BaseModel):
    query: str
    strategy: str = "least_loaded"   # "least_loaded" | "random" | "round_robin"


# ── Helper ───────────────────────────────────────────────────────────────────

def _get_live_nodes() -> List[dict]:
    """Returns all nodes that have sent a heartbeat within the timeout window."""
    now = time.time()
    return [
        n for n in _swarm_nodes.values()
        if now - n.get("last_seen", 0) < _HEARTBEAT_TIMEOUT
    ]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/swarm/register")
async def register_node(reg: NodeRegistration):
    """Child node registers itself when it boots up."""
    _swarm_nodes[reg.node_id] = {
        "node_id": reg.node_id,
        "label": reg.label,
        "host": reg.host,
        "port": reg.port,
        "parent_id": reg.parent_id,
        "load": 0.0,
        "requests_served": 0,
        "registered_at": time.time(),
        "last_seen": time.time(),
        "status": "alive",
    }
    print(f"🔌 [CloudBroker] 새 노드 등록: [{reg.node_id[:8]}] '{reg.label}' @ {reg.host}:{reg.port}")
    return {"status": "registered", "node_id": reg.node_id}


@app.post("/swarm/heartbeat")
async def node_heartbeat(hb: NodeHeartbeat):
    """Child node sends periodic heartbeat to stay alive in the registry."""
    node = _swarm_nodes.get(hb.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not registered")
    node["last_seen"] = time.time()
    node["load"] = hb.load
    node["requests_served"] = hb.requests_served
    node["status"] = "alive"
    return {"status": "ok"}


@app.get("/swarm/nodes")
async def list_nodes():
    """Returns all currently-live swarm nodes."""
    live = _get_live_nodes()
    return {
        "total": len(live),
        "nodes": live,
    }


@app.post("/swarm/dispatch")
async def dispatch_query(req: DispatchRequest):
    """
    Routes a query to the best available child node based on strategy.
    Returns the node's response or an error if no node is available.
    """
    live = _get_live_nodes()
    if not live:
        raise HTTPException(status_code=503, detail="No live child nodes available")

    # Select target node
    if req.strategy == "least_loaded":
        target = min(live, key=lambda n: n["load"])
    elif req.strategy == "random":
        target = random.choice(live)
    else:
        # round_robin: pick the one with fewest requests served
        target = min(live, key=lambda n: n["requests_served"])

    url = f"http://{target['host']}:{target['port']}/query"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={"query": req.query})
            resp.raise_for_status()
            result = resp.json()
            # Update request count on the registry entry
            _swarm_nodes[target["node_id"]]["requests_served"] += 1
            return {"node_id": target["node_id"], "label": target["label"], "response": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Child node failed: {e}")


@app.delete("/swarm/node/{node_id}")
async def remove_node(node_id: str):
    """De-registers a node — called by the provisioner when culling."""
    if node_id not in _swarm_nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    del _swarm_nodes[node_id]
    print(f"🗑️  [CloudBroker] 노드 제거: {node_id[:8]}")
    return {"status": "removed", "node_id": node_id}


@app.get("/swarm/status")
async def swarm_status():
    """Health dashboard for the entire swarm."""
    live = _get_live_nodes()
    total_requests = sum(n.get("requests_served", 0) for n in live)
    avg_load = sum(n.get("load", 0) for n in live) / max(1, len(live))
    return {
        "live_nodes": len(live),
        "total_registered": len(_swarm_nodes),
        "total_requests_served": total_requests,
        "avg_load": round(avg_load, 3),
        "nodes": [
            {
                "node_id": n["node_id"][:8],
                "label": n["label"],
                "load": n["load"],
                "requests_served": n["requests_served"],
                "uptime_s": round(time.time() - n["registered_at"]),
            }
            for n in live
        ],
    }


@app.get("/")
async def root():
    return {"service": "FZA Cloud Broker v16.0", "live_nodes": len(_get_live_nodes())}
