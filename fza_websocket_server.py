import asyncio
import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from fza_event_bus import bus
from main_fza_system import FZAManager

# Global State
manager = None
ws_manager = None
main_loop = None

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        cid = str(uuid.uuid4())
        self.active_connections[cid] = websocket
        return cid

    def disconnect(self, cid: str):
        if cid in self.active_connections:
            del self.active_connections[cid]

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            try:
                await connection.send_json(message)
            except Exception:
                pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, ws_manager, main_loop
    
    print("🚀 [Server] Initializing FZA Engine...")
    manager = FZAManager(use_local=True)
    ws_manager = ConnectionManager()
    main_loop = asyncio.get_running_loop()
    
    def on_bus_event(payload):
        if main_loop and not main_loop.is_closed():
            # Use call_soon_threadsafe so synchronous threads can safely push to WS
            main_loop.call_soon_threadsafe(
                lambda: main_loop.create_task(ws_manager.broadcast(payload))
            )
            
    bus.add_listener(on_bus_event)
    print("📡 [WebSocket] Event Bus listener attached.")
    
    yield
    print("🛑 [Server] Shutting down FZA Engine...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    cid = await ws_manager.connect(websocket)
    try:
        # Send initial state
        state = {
            "root_facts": {k: v for k, v in manager.bridge.user_profile.items() if not k.startswith("_")},
            "leaf_memories": manager.bridge.user_profile.get("_memories", []),
            "adapters": manager.router.adapter_count if getattr(manager, 'router', None) else 0
        }
        await websocket.send_json({"type": "engine_state", "data": state})
        
        while True:
            data = await websocket.receive_text()
            try:
                req = json.loads(data)
                action = req.get("action")
                
                if action == "chat":
                    msg = req.get("message", "")
                    
                    # Log message receipt
                    await ws_manager.broadcast({"type": "user_message", "data": msg})
                    
                    # Process in a thread so we don't block the ASGI loop
                    loop = asyncio.get_running_loop()
                    reply = await loop.run_in_executor(None, manager.chat_and_remember, msg)
                    
                    await ws_manager.broadcast({"type": "engine_reply", "data": reply["reply"] if isinstance(reply, dict) else reply})
                    
            except json.JSONDecodeError:
                print(f"⚠️ [WebSocket] Invalid JSON received.")
                
    except WebSocketDisconnect:
        ws_manager.disconnect(cid)

@app.get("/state")
async def get_state():
    return {
        "status": "online",
        "root_facts": {k: v for k, v in manager.bridge.user_profile.items() if not k.startswith("_")},
        "leaf_memories": manager.bridge.user_profile.get("_memories", []),
        "rag_nodes": len(manager.rag.index) if getattr(manager, 'rag', None) else 0,
        "adapters": manager.router.adapter_count if getattr(manager, 'router', None) else 0
    }

if __name__ == "__main__":
    uvicorn.run("fza_websocket_server:app", host="0.0.0.0", port=8000, reload=False)
