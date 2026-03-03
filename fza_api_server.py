"""
fza_api_server.py — FZA Web API Bridge
========================================
This module wraps the FZA Local Engine in a FastAPI application, exposing
a WebSocket endpoint that the React frontend can connect to.

It maps browser WebSocket messages → FZALocalEngine.process() 
and FZA Event Bus events → WebSocket messages back to the UI.

Usage:
    pip install fastapi uvicorn
    uvicorn fza_api_server:app --host 0.0.0.0 --port 8000
"""

import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Set

# FZA Core Imports
from fza_local_engine import FZALocalEngine
from fza_event_bus import bus

app = FastAPI(title="FZA Core API Bridge")

# Completely permissive CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebSocketManager:
    """Manages active websocket connections from the React UI."""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"🔌 [API] 클라이언트 연결 (총 {len(self.active_connections)}명)")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"🔌 [API] 클라이언트 연결 해제 (총 {len(self.active_connections)}명)")

    async def broadcast(self, message: dict):
        # Fire-and-forget broadcast to all connected UIs
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = WebSocketManager()
engine: FZALocalEngine = None


@app.on_event("startup")
async def startup_event():
    """Initialize the FZA Engine when the web server starts."""
    global engine
    
    # Check if a model name is provided in env, else use default
    model_name = os.environ.get("FZA_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    engine = FZALocalEngine(model_name=model_name)
    
    # Wire FZA Event Bus events to the WebSocket Manager
    # This allows the UI to see internal FZA thoughts as they happen
    
    def _forward_event(event_type: str):
        def _handler(data):
            msg = {"type": "fza_event", "event_type": event_type, "data": data}
            # Use asyncio.create_task to safely call async method from sync callback
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(manager.broadcast(msg))
            except RuntimeError:
                pass # Event loop not running locally yet
        return _handler

    # List of interesting deep-brain events to show in the UI
    interesting_events = [
        "reflex_fire", "micro_reflex_fire", 
        "graph_node_added", "graph_node_updated",
        "sleep_spindle_start", "lore_adapter_saved",
        "lobby_registered", "api_error",
        "DEGRADATION_CRITICAL", "apoptosis_started"
    ]
    for ev in interesting_events:
        bus.on(ev, _forward_event(ev))
        
    print(f"🚀 [API] FZA 엔진 로드 완료. ws://localhost:8000/ws/chat 대기 중.")


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main duplex connection for the React UI.
    Receives user messages, passes them to FZA, and returns responses.
    """
    await manager.connect(websocket)
    try:
        # Send initial state
        await _send_engine_state()
        
        while True:
            data_raw = await websocket.receive_text()
            try:
                msg = json.loads(data_raw)
                user_text = msg.get("message", "")
            except:
                user_text = data_raw
                
            print(f"👤 [Web] 입력: {user_text}")
            
            # Echo message to UI to show we got it
            await manager.broadcast({"type": "user_message", "data": user_text})
            # Send immediate ACK so UI knows it's thinking
            await manager.broadcast({"type": "status", "status": "thinking"})
            
            try:
                # To stream tokens, we need to capture stdout or modify the engine.
                # For now, we'll run the engine in a thread and simulate token streaming
                # by capturing the final response and chunking it (or intercepting if possible).
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, engine.process, user_text)
                
                # Stream the response chunk by chunk (simulate streaming if not natively hooked)
                # Production FZA would yield generator chunks here.
                words = response.response_text.split(" ")
                for i, word in enumerate(words):
                    await manager.broadcast({
                        "type": "token",
                        "data": {"text": word + (" " if i < len(words)-1 else "")}
                    })
                    await asyncio.sleep(0.02)  # 20ms delay per word for visual effect
                
                # Send final reply marker
                await manager.broadcast({
                    "type": "engine_reply",
                    "data": response.response_text
                })
                
                # Update UI state with new memory/adapters
                await _send_engine_state()
                
            except Exception as e:
                print(f"❌ [API] 엔진 에러: {e}")
                import traceback
                traceback.print_exc()
                await manager.broadcast({"type": "chat_error", "error": str(e)})
                
            # Send idle status
            await manager.broadcast({"type": "status", "status": "idle"})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def _send_engine_state():
    """Reads the current FZA state and sends it to the UI (The Vault & Stats)."""
    if not engine: return
    
    root_facts = {}
    leaf_mems = []
    
    # Safely extract from memory graph
    try:
        if engine.memory_graph:
            root_facts = engine.memory_graph.root_nodes
            # Get last 5 episodic events as leaf memories
            if hasattr(engine.memory_graph, 'episodic_log'):
                leaf_mems = [f"[{e['timestamp']}] {e['event_type']}" for e in engine.memory_graph.episodic_log[-5:]]
    except:
        pass
        
    # Count adapters
    adapters = 0
    try:
        import os
        from fza_health_monitor import HealthMonitor
        hm = HealthMonitor(auto_check=False)
        adapters = hm.count_adapters()
    except:
        pass

    state_payload = {
        "type": "engine_state",
        "data": {
            "root_facts": root_facts,
            "leaf_memories": leaf_mems,
            "adapters": adapters
        }
    }
    await manager.broadcast(state_payload)
