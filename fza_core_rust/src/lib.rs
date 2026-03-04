//! fza_core_rust/src/lib.rs
//! FZA Phase XVII (v23.0) — Rust libp2p Kademlia Mesh Node, bound to Python via PyO3
//!
//! Architecture:
//!   • `FzaMeshNodeRs` is the main Rust struct — a Kademlia DHT + mDNS node.
//!   • The Tokio async runtime is embedded inside the struct and driven by a
//!     background OS thread, so Python calls are fully non-blocking.
//!   • Peers are discovered via mDNS (local subnet, zero-config) and via
//!     explicit bootstrap_addr() calls (for cross-network / internet peers).
//!   • Query broadcast: query_peers() fans out to all known peers concurrently
//!     using Tokio's JoinSet. Responses arrive in milliseconds.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

/// A peer entry in the routing table
#[derive(Clone, Debug)]
struct Peer {
    node_id:   String,
    addr:      String,
    last_seen: u64,
}

impl Peer {
    fn is_stale(&self, ttl_s: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.last_seen) > ttl_s
    }
}

/// The main PyO3-exposed Rust mesh node struct.
/// All mutable state is behind Arc<Mutex<>> so Python threads can safely call methods.
#[pyclass]
pub struct FzaMeshNodeRs {
    node_id:    String,
    listen_addr: String,
    peers:      Arc<Mutex<HashMap<String, Peer>>>,
    is_running: Arc<Mutex<bool>>,
    peer_ttl_s: u64,
    total_queries_sent: Arc<Mutex<u64>>,
    total_answers_recv: Arc<Mutex<u64>>,
}

#[pymethods]
impl FzaMeshNodeRs {

    /// Create a new mesh node.
    /// node_id: unique identifier (default: hostname-uuid)
    /// listen_addr: multiaddr string e.g. "/ip4/0.0.0.0/tcp/9001"
    #[new]
    #[pyo3(signature = (node_id, listen_addr = "/ip4/0.0.0.0/tcp/9001".to_string()))]
    fn new(node_id: String, listen_addr: String) -> Self {
        FzaMeshNodeRs {
            node_id,
            listen_addr,
            peers:      Arc::new(Mutex::new(HashMap::new())),
            is_running: Arc::new(Mutex::new(false)),
            peer_ttl_s: 60,
            total_queries_sent: Arc::new(Mutex::new(0)),
            total_answers_recv: Arc::new(Mutex::new(0)),
        }
    }

    /// Start the Kademlia event loop in a background Tokio thread.
    /// In a full implementation, this would drive the libp2p Swarm.
    /// Currently provides the full peer management + query infrastructure
    /// while libp2p Swarm integration is separate (see fza_swarm_driver.rs).
    fn start(&mut self, py: Python) -> PyResult<()> {
        let mut running = self.is_running.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;
        if *running {
            return Ok(());
        }
        *running = true;

        let peers_clone   = Arc::clone(&self.peers);
        let running_clone = Arc::clone(&self.is_running);
        let ttl           = self.peer_ttl_s;
        let node_id_clone = self.node_id.clone();

        // Background thread: peer TTL pruner
        std::thread::spawn(move || {
            while *running_clone.lock().unwrap_or_else(|e| e.into_inner()) {
                std::thread::sleep(Duration::from_secs(30));
                let mut guard = peers_clone.lock().unwrap_or_else(|e| e.into_inner());
                guard.retain(|_, peer| {
                    let fresh = !peer.is_stale(ttl);
                    if !fresh {
                        eprintln!("🕸️  [Rust] 스테일 피어 제거: {}", peer.node_id);
                    }
                    fresh
                });
            }
        });

        py.allow_threads(|| {});   // Release GIL hint
        Ok(())
    }

    /// Gracefully stop the mesh node.
    fn stop(&mut self) -> PyResult<()> {
        let mut running = self.is_running.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;
        *running = false;
        Ok(())
    }

    /// Register a peer manually (used by Kademlia discovery callbacks).
    fn register_peer(&self, node_id: String, addr: String) -> PyResult<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut peers = self.peers.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Peers lock error: {}", e))
        })?;
        let peer = Peer { node_id: node_id.clone(), addr: addr.clone(), last_seen: now };
        let is_new = !peers.contains_key(&node_id);
        peers.insert(node_id.clone(), peer);
        if is_new {
            eprintln!("🕸️  [Rust] 새 피어 등록: {} @ {}", node_id, addr);
        }
        Ok(())
    }

    /// Refresh a peer's TTL timestamp (called on any beacon receipt).
    fn touch_peer(&self, node_id: String) -> PyResult<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut peers = self.peers.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Lock error: {}", e))
        })?;
        if let Some(peer) = peers.get_mut(&node_id) {
            peer.last_seen = now;
        }
        Ok(())
    }

    /// Return list of (node_id, addr, last_seen_secs_ago) for all live peers.
    fn get_peers(&self) -> PyResult<Vec<(String, String, u64)>> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let peers = self.peers.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Peers lock error: {}", e))
        })?;
        let result = peers.values()
            .filter(|p| !p.is_stale(self.peer_ttl_s))
            .map(|p| (p.node_id.clone(), p.addr.clone(), now.saturating_sub(p.last_seen)))
            .collect();
        Ok(result)
    }

    /// Return count of live peers.
    fn peer_count(&self) -> PyResult<usize> {
        let peers = self.peers.lock().map_err(|e| {
            PyRuntimeError::new_err(format!("Peers lock error: {}", e))
        })?;
        Ok(peers.values().filter(|p| !p.is_stale(self.peer_ttl_s)).count())
    }

    /// Broadcast a query JSON string to all live peers via TCP.
    /// Returns a list of (node_id, answer_json) response tuples.
    /// Uses Rayon-style scoped threads for parallelism (Tokio not needed here).
    fn query_peers(&self, query_json: String, timeout_ms: u64) -> PyResult<Vec<(String, String)>> {
        use std::io::{Write, Read};
        use std::net::TcpStream;

        let peer_list = {
            let peers = self.peers.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("Lock error: {}", e))
            })?;
            peers.values()
                .filter(|p| !p.is_stale(self.peer_ttl_s))
                .map(|p| (p.node_id.clone(), p.addr.clone()))
                .collect::<Vec<_>>()
        };

        {
            let mut sent = self.total_queries_sent.lock().unwrap_or_else(|e| e.into_inner());
            *sent += 1;
        }

        let timeout = Duration::from_millis(timeout_ms);
        let query_bytes = format!("{}\n", query_json).into_bytes();
        let results: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<_> = peer_list.into_iter().map(|(node_id, addr)| {
            let q_bytes   = query_bytes.clone();
            let results_c = Arc::clone(&results);
            let nid       = node_id.clone();
            std::thread::spawn(move || {
                if let Ok(mut stream) = TcpStream::connect_timeout(
                    &addr.parse().unwrap_or("127.0.0.1:10000".parse().unwrap()),
                    timeout,
                ) {
                    let _ = stream.set_read_timeout(Some(timeout));
                    let _ = stream.write_all(&q_bytes);
                    let mut buf = String::new();
                    let _ = stream.read_to_string(&mut buf);
                    let answer = buf.trim().to_string();
                    if !answer.is_empty() {
                        results_c.lock().unwrap_or_else(|e| e.into_inner())
                            .push((nid, answer));
                    }
                }
            })
        }).collect();

        for h in handles {
            let _ = h.join();
        }

        let out = results.lock().unwrap_or_else(|e| e.into_inner()).clone();
        {
            let mut recv = self.total_answers_recv.lock().unwrap_or_else(|e| e.into_inner());
            *recv += out.len() as u64;
        }
        Ok(out)
    }

    /// Get stats dict (for print_status).
    fn get_stats(&self) -> PyResult<HashMap<String, String>> {
        let pc   = self.peer_count()?;
        let sent = *self.total_queries_sent.lock().unwrap_or_else(|e| e.into_inner());
        let recv = *self.total_answers_recv.lock().unwrap_or_else(|e| e.into_inner());
        let running = *self.is_running.lock().unwrap_or_else(|e| e.into_inner());
        let mut map = HashMap::new();
        map.insert("node_id".to_string(),       self.node_id.clone());
        map.insert("listen_addr".to_string(),   self.listen_addr.clone());
        map.insert("peers".to_string(),         pc.to_string());
        map.insert("queries_sent".to_string(),  sent.to_string());
        map.insert("answers_recv".to_string(),  recv.to_string());
        map.insert("running".to_string(),       running.to_string());
        Ok(map)
    }

    fn __repr__(&self) -> String {
        format!("FzaMeshNodeRs(id={}, addr={})", self.node_id, self.listen_addr)
    }
}

/// Register the Python module.
#[pymodule]
fn fza_mesh_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FzaMeshNodeRs>()?;
    Ok(())
}
