import threading
import json

class EventBus:
    """
    A thread-safe singleton EventBus used to pipe internal FZA events
    out to active WebSocket clients.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.listeners = []
            cls._lock = threading.Lock()
        return cls._instance
        
    def add_listener(self, func):
        with self._lock:
            if func not in self.listeners:
                self.listeners.append(func)
            
    def remove_listener(self, func):
        with self._lock:
            if func in self.listeners:
                self.listeners.remove(func)
                
    def emit(self, event_type: str, data: dict = None):
        if data is None:
            data = {}
        payload = {"type": event_type, "data": data}
        with self._lock:
            # Copy listeners to avoid mutation during iteration
            current_listeners = list(self.listeners)
        
        for l in current_listeners:
            try:
                l(payload)
            except Exception as e:
                print(f"⚠️ [EventBus] Listener error: {e}")

# Global singleton
bus = EventBus()
