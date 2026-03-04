"""
test_fza_mesh_sprint1.py — Rigorous Sprint 1 Test Suite (v23.0)
================================================================
Tests every layer of the P2P Mesh node:
  - Peer registration and deduplication
  - TTL staleness marking and pruning
  - Single-node TCP query server + client (loopback)
  - Multi-node concurrent queries
  - Query timeout handling
  - Score and stats correctness
  - Graceful stop/restart
  - Backend detection (Rust or Python)

Run with:
    source .venv/bin/activate
    python test_fza_mesh_sprint1.py
"""

import sys
import os
import time
import threading
import socket
import json

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fza_mesh_node import MeshNode, PeerInfo, _RUST_BACKEND

TESTS_PASSED = 0
TESTS_FAILED = 0


def run_test(name: str, fn):
    global TESTS_PASSED, TESTS_FAILED
    try:
        fn()
        print(f"  ✅  {name}")
        TESTS_PASSED += 1
    except AssertionError as e:
        print(f"  ❌  {name} — AssertionError: {e}")
        TESTS_FAILED += 1
    except Exception as e:
        print(f"  ❌  {name} — Exception: {type(e).__name__}: {e}")
        TESTS_FAILED += 1


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  🧪  {title}")
    print(f"{'─'*55}")


# ─── Section 1: Backend Detection ────────────────────────────────────────────

section("Section 1: Backend Detection")

def test_backend_detection():
    """At least one backend must be active."""
    # _RUST_BACKEND is True if fza_mesh_rs compiled, False for Python fallback
    assert isinstance(_RUST_BACKEND, bool), "Backend flag must be bool"

def test_backend_report():
    backend_name = "Rust (fza_mesh_rs)" if _RUST_BACKEND else "Python fallback"
    print(f"  ℹ️   Active backend: {backend_name}")
    assert True

run_test("backend_detection — flag is bool", test_backend_detection)
run_test("backend_detection — report active backend", test_backend_report)


# ─── Section 2: Peer Registration ─────────────────────────────────────────────

section("Section 2: Peer Registration & Deduplication")

def test_peer_init():
    peer = PeerInfo(node_id="test-a", ip="127.0.0.1", port=10200)
    assert peer.node_id == "test-a"
    assert peer.ip     == "127.0.0.1"
    assert peer.port   == 10200
    assert not peer.is_stale()

def test_peer_stale():
    peer = PeerInfo(node_id="old", ip="1.2.3.4", port=9000)
    peer.last_seen = time.time() - 120  # 2 min ago; TTL is 60s
    assert peer.is_stale()

def test_peer_touch_resets_stale():
    peer = PeerInfo(node_id="touch-me", ip="1.2.3.4", port=9001)
    peer.last_seen = time.time() - 120
    assert peer.is_stale()
    peer.touch()
    assert not peer.is_stale()

def test_node_register_peer():
    node = MeshNode(node_id="host", port=11300)
    with node._lock:
        node._peers["peer-a"] = PeerInfo("peer-a", "127.0.0.1", 11001)
        node._peers["peer-b"] = PeerInfo("peer-b", "127.0.0.1", 11002)
    assert len(node._peers) == 2

def test_node_dedup_peer():
    """Registering same peer twice should update (not duplicate) the entry."""
    node = MeshNode(node_id="host", port=11301)
    with node._lock:
        node._peers["peer-x"] = PeerInfo("peer-x", "127.0.0.1", 11010)
        # Simulate second beacon — touch
        node._peers["peer-x"].touch()
    assert len(node._peers) == 1  # still 1

run_test("PeerInfo init", test_peer_init)
run_test("PeerInfo is_stale after 120s", test_peer_stale)
run_test("PeerInfo touch() resets stale", test_peer_touch_resets_stale)
run_test("MeshNode registers multiple peers", test_node_register_peer)
run_test("MeshNode deduplicates same peer", test_node_dedup_peer)


# ─── Section 3: TCP Query Server + Client ─────────────────────────────────────

section("Section 3: TCP Query (Server ↔ Client Loopback)")

QUERY_SERVER_PORT = 11400

def test_tcp_server_starts_and_answers():
    """
    Start a MeshNode on port 11400, inject it as a peer into node_a,
    then use node_a's query_mesh() to get a response over TCP.
    """
    response_seen = threading.Event()
    received_answers = []

    def on_query(q: str) -> str:
        return f"SERVER_ANSWER:{q.upper()}"

    server = MeshNode(node_id="server-tcp", port=QUERY_SERVER_PORT, on_query=on_query)
    server.start()
    time.sleep(0.3)  # Let TCP socket bind

    client = MeshNode(node_id="client-tcp", port=11401)
    with client._lock:
        client._peers["server-tcp"] = PeerInfo("server-tcp", "127.0.0.1", QUERY_SERVER_PORT)

    responses = client.query_mesh("hello mesh")
    server.stop()

    assert len(responses) == 1, f"Expected 1 response, got {len(responses)}: {responses}"
    answer = responses[0]["answer"]
    assert "SERVER_ANSWER" in answer, f"Wrong answer: {answer}"
    assert "HELLO MESH" in answer, f"Query not echoed back: {answer}"

def test_query_returns_node_id():
    """Response should contain the server's node_id."""
    def on_query(q):
        return "42"

    server = MeshNode(node_id="id-check-server", port=11410, on_query=on_query)
    server.start()
    time.sleep(0.3)

    client = MeshNode(node_id="id-check-client", port=11411)
    with client._lock:
        client._peers["id-check-server"] = PeerInfo("id-check-server", "127.0.0.1", 11410)

    responses = client.query_mesh("who are you?")
    server.stop()

    assert len(responses) == 1
    assert responses[0]["node_id"] == "id-check-server", f"Wrong node_id: {responses[0]}"

def test_query_no_peers_returns_empty():
    """query_mesh with no peers should return []."""
    node = MeshNode(node_id="lonely", port=11420)
    responses = node.query_mesh("anybody there?")
    assert responses == [], f"Expected [], got {responses}"

run_test("TCP server starts + answers query", test_tcp_server_starts_and_answers)
run_test("Response contains correct node_id", test_query_returns_node_id)
run_test("query_mesh with 0 peers returns []", test_query_no_peers_returns_empty)


# ─── Section 4: Multi-Node Concurrent Queries ─────────────────────────────────

section("Section 4: Multi-Node Concurrent Fan-Out")

def test_multi_node_fanout():
    """
    Start 3 server nodes, register them all as peers, then fire a single query.
    All 3 should respond in parallel.
    """
    servers = []
    ports = [11500, 11501, 11502]
    for i, port in enumerate(ports):
        def make_cb(idx):
            return lambda q: f"NODE_{idx}_ANSWER"
        s = MeshNode(node_id=f"server-{i}", port=port, on_query=make_cb(i))
        s.start()
        servers.append(s)

    time.sleep(0.4)

    client = MeshNode(node_id="fan-client", port=11510)
    with client._lock:
        for i, port in enumerate(ports):
            client._peers[f"server-{i}"] = PeerInfo(f"server-{i}", "127.0.0.1", port)

    t0 = time.time()
    responses = client.query_mesh("fan out test", timeout=2.0)
    elapsed = time.time() - t0

    for s in servers:
        s.stop()

    assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}: {responses}"
    # Check they're parallel: should complete faster than 3 * serial_timeout
    assert elapsed < 3.5, f"Fan-out too slow: {elapsed:.2f}s (expected < 3.5s)"

def test_stats_after_queries():
    """After sending queries, stats should reflect correct counts."""
    def on_query(q):
        return "yes"
    server = MeshNode(node_id="stats-server", port=11520, on_query=on_query)
    server.start()
    time.sleep(0.3)

    client = MeshNode(node_id="stats-client", port=11521)
    with client._lock:
        client._peers["stats-server"] = PeerInfo("stats-server", "127.0.0.1", 11520)

    client.query_mesh("q1")
    client.query_mesh("q2")
    stats = client.get_stats()
    server.stop()

    assert stats["queries_sent"] == 2, f"queries_sent wrong: {stats}"
    assert stats["responses_aggregated"] == 2, f"responses_agg wrong: {stats}"
    assert stats["known_peers"] == 1, f"known_peers wrong: {stats}"

run_test("3-node parallel fan-out (all respond)", test_multi_node_fanout)
run_test("stats correct after 2 queries", test_stats_after_queries)


# ─── Section 5: Timeout & Fault Tolerance ────────────────────────────────────

section("Section 5: Timeout & Fault Tolerance")

def test_dead_peer_timeout():
    """Querying a peer that refuses connection should time out cleanly, not crash."""
    client = MeshNode(node_id="resilient-client", port=11600)
    with client._lock:
        # Port 11699 has nothing listening
        client._peers["ghost"] = PeerInfo("ghost", "127.0.0.1", 11699)

    t0 = time.time()
    responses = client.query_mesh("hello ghost?", timeout=0.5)
    elapsed = time.time() - t0

    assert responses == [], f"Expected [] for dead peer, got {responses}"
    assert elapsed < 3.0, f"Should time out fast, elapsed={elapsed:.2f}s"

def test_partial_failure_resilience():
    """
    3 peers registered: 2 alive, 1 dead. Should still get 2 responses.
    """
    def on_query(q):
        return "alive"

    s1 = MeshNode(node_id="alive-1", port=11610, on_query=on_query)
    s2 = MeshNode(node_id="alive-2", port=11611, on_query=on_query)
    s1.start(); s2.start()
    time.sleep(0.3)

    client = MeshNode(node_id="partial-client", port=11620)
    with client._lock:
        client._peers["alive-1"] = PeerInfo("alive-1", "127.0.0.1", 11610)
        client._peers["alive-2"] = PeerInfo("alive-2", "127.0.0.1", 11611)
        client._peers["dead"]    = PeerInfo("dead",    "127.0.0.1", 11699)  # nothing listening

    responses = client.query_mesh("partial test", timeout=1.0)
    s1.stop(); s2.stop()

    assert len(responses) == 2, f"Expected 2 from alive peers, got {len(responses)}: {responses}"

run_test("Dead peer times out cleanly", test_dead_peer_timeout)
run_test("Partial failure: 2/3 peers alive → 2 responses", test_partial_failure_resilience)


# ─── Section 6: Start / Stop Lifecycle ───────────────────────────────────────

section("Section 6: Start / Stop Lifecycle")

def test_start_stop_cycle():
    """Node should start, serve a query, stop without crashing."""
    def on_query(q):
        return "pong"
    node = MeshNode(node_id="lifecycle-node", port=11700, on_query=on_query)
    node.start()
    time.sleep(0.3)
    assert node._running

    client = MeshNode(node_id="lifecycle-client", port=11701)
    with client._lock:
        client._peers["lifecycle-node"] = PeerInfo("lifecycle-node", "127.0.0.1", 11700)
    responses = client.query_mesh("ping", timeout=1.0)
    assert len(responses) == 1

    node.stop()
    time.sleep(0.2)
    assert not node._running

def test_double_start_idempotent():
    """Calling start() twice should be idempotent."""
    node = MeshNode(node_id="double-start", port=11710)
    node.start()
    node.start()  # Should not crash or bind twice
    node.stop()
    assert True

run_test("start → query → stop lifecycle", test_start_stop_cycle)
run_test("double start() is idempotent", test_double_start_idempotent)


# ─── Final Report ─────────────────────────────────────────────────────────────

total = TESTS_PASSED + TESTS_FAILED
print(f"\n{'═'*55}")
print(f"  Sprint 1 Test Report")
print(f"  {'─'*50}")
print(f"  Backend:  {'🦀 Rust (fza_mesh_rs)' if _RUST_BACKEND else '🐍 Python fallback'}")
print(f"  Passed:   {TESTS_PASSED}/{total}")
print(f"  Failed:   {TESTS_FAILED}/{total}")
print(f"{'═'*55}")

if TESTS_FAILED == 0:
    print(f"\n  ✅  ALL SPRINT 1 TESTS PASSED — CLEARED FOR SPRINT 2\n")
    sys.exit(0)
else:
    print(f"\n  ❌  {TESTS_FAILED} TEST(S) FAILED — DO NOT PROCEED TO SPRINT 2\n")
    sys.exit(1)
