"""
Tests for the worker node changes:
- network.py: send_bytes / receive_bytes aliases
- worker.py: activation wire format (send/receive round-trip)
- worker.py: KVCacheConfig wiring through load_weights
- worker.py: run() loop dispatch (SHUTDOWN, FLUSH_CACHE, LOAD_LAYER, EXECUTE_LAYER)

Tests run entirely in-process using pairs of connected sockets – no real
root node is required.
"""

import sys
import os
import socket
import struct
import threading
import tempfile
import shutil
import types

import numpy as np

# Make airllm and distributed_llama_python importable
_here = os.path.dirname(__file__)
sys.path.insert(0, _here)

# ---------------------------------------------------------------------------
# Lazy imports (after path setup)
# ---------------------------------------------------------------------------
from distributed_llama_python.network import NetworkClient
from distributed_llama_python.control_protocol import (
    ControlMessage, OpType,
)
from distributed_llama_python.worker import Worker
from airllm.kv_cache import KVCacheConfig, DiskKVCacheManager


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_socketpair():
    """Return two connected TCP sockets (loopback)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(('127.0.0.1', port))
    conn, _ = srv.accept()
    srv.close()
    return cli, conn


def _make_network_pair():
    """Return two NetworkClient instances sharing a socketpair."""
    cli_sock, srv_sock = _make_socketpair()

    def _make(sock):
        net = NetworkClient.__new__(NetworkClient)
        net.host = '127.0.0.1'
        net.port = 0
        net.socket = sock
        net.sent_bytes = 0
        net.recv_bytes = 0
        return net

    return _make(cli_sock), _make(srv_sock)


def _build_worker(cli_net, compress='none', kv_mgr=None, assigned_layers=None):
    """
    Build a Worker with a mock inference engine and the given NetworkClient.
    The mock engine returns the input activations unchanged (identity).
    """
    class _MockEngine:
        def __init__(self, kv_cache_manager):
            self.kv_cache_manager = kv_cache_manager
            self.header = types.SimpleNamespace(dim=64, n_layers=2)

        def execute_layer(self, layer_id, x, pos=0, kv_cache=None,
                          kv_cache_manager=None):
            return x, None

        def load_layer(self, layer_id, prefetch_next=False):
            pass

        def cleanup(self):
            pass

    w = Worker.__new__(Worker)
    w.host = '127.0.0.1'
    w.port = 0
    w.model_path = None
    w.kv_cache_config = None
    w.compress_activations_method = compress
    w.network = cli_net
    w.net_config = None
    w.node_config = None
    w.pipes = {}
    w.inference_engine = _MockEngine(kv_cache_manager=kv_mgr)
    w.assigned_layers = assigned_layers if assigned_layers is not None else [0, 1]
    return w


def _send_ctrl(net, op_type, layer_id=0, offset=0, size=0, seq_len=0):
    msg = ControlMessage(op_type=op_type, layer_id=layer_id,
                         offset=offset, size=size, seq_len=seq_len)
    net.write(msg.to_bytes())


# ── network.py aliases ────────────────────────────────────────────────────────

def test_network_aliases():
    """send_bytes / receive_bytes should be aliases for write / read."""
    print("\n=== network: send_bytes / receive_bytes aliases ===")

    cli, srv = _make_network_pair()
    payload = b'\x01\x02\x03\x04'

    def _send():
        cli.send_bytes(payload)

    t = threading.Thread(target=_send, daemon=True)
    t.start()
    received = srv.receive_bytes(len(payload))
    t.join(timeout=2)

    assert received == payload, f"expected {payload!r}, got {received!r}"
    print("  ✓ send_bytes / receive_bytes aliases work")

    cli.socket.close()
    srv.socket.close()


# ── activation wire format ────────────────────────────────────────────────────

def test_activation_roundtrip_q80():
    """send_activations → receive_activations round-trip with q80 compression."""
    print("\n=== activation wire format: q80 round-trip ===")

    cli_net, srv_net = _make_network_pair()
    sender   = _build_worker(cli_net, compress='q80')
    receiver = _build_worker(srv_net, compress='q80')

    x_orig = np.random.randn(1, 128).astype(np.float32)
    shape  = x_orig.shape

    def _send():
        sender.send_activations(x_orig)

    t = threading.Thread(target=_send, daemon=True)
    t.start()
    x_recv = receiver.receive_activations(expected_shape=shape)
    t.join(timeout=2)

    assert x_recv.shape == shape, f"shape mismatch: {x_recv.shape}"
    mse = float(np.mean((x_orig - x_recv) ** 2))
    print(f"  MSE after q80 round-trip: {mse:.6f}")
    assert mse < 0.01, f"MSE too large: {mse}"
    print("  ✓ q80 activation round-trip passed")

    cli_net.socket.close()
    srv_net.socket.close()


def test_activation_roundtrip_none():
    """send_activations → receive_activations with no compression (exact)."""
    print("\n=== activation wire format: no-compression round-trip ===")

    cli_net, srv_net = _make_network_pair()
    sender   = _build_worker(cli_net, compress='none')
    receiver = _build_worker(srv_net, compress='none')

    x_orig = np.random.randn(2, 64).astype(np.float32)
    shape  = x_orig.shape

    def _send():
        sender.send_activations(x_orig)

    t = threading.Thread(target=_send, daemon=True)
    t.start()
    x_recv = receiver.receive_activations(expected_shape=shape)
    t.join(timeout=2)

    assert np.array_equal(x_orig, x_recv), "uncompressed tensors must be identical"
    print("  ✓ no-compression activation round-trip passed")

    cli_net.socket.close()
    srv_net.socket.close()


# ── run() loop dispatch ───────────────────────────────────────────────────────

def test_run_shutdown():
    """SHUTDOWN message causes run() to exit cleanly."""
    print("\n=== run() loop: SHUTDOWN ===")

    cli_net, srv_net = _make_network_pair()
    worker = _build_worker(cli_net)

    def _root():
        _send_ctrl(srv_net, OpType.SHUTDOWN)
        srv_net.read_ack()          # ACK from worker after SHUTDOWN

    t = threading.Thread(target=_root, daemon=True)
    t.start()
    worker.run()
    t.join(timeout=3)
    assert not t.is_alive(), "root thread must finish after SHUTDOWN"
    print("  ✓ SHUTDOWN causes run() to exit")

    cli_net.socket.close()
    srv_net.socket.close()


def test_run_load_layer():
    """LOAD_LAYER triggers layer pre-loading then returns ACK."""
    print("\n=== run() loop: LOAD_LAYER ===")

    cli_net, srv_net = _make_network_pair()
    worker = _build_worker(cli_net)

    def _root():
        _send_ctrl(srv_net, OpType.LOAD_LAYER, layer_id=0)
        srv_net.read_ack()          # ACK from LOAD_LAYER
        _send_ctrl(srv_net, OpType.SHUTDOWN)
        srv_net.read_ack()

    t = threading.Thread(target=_root, daemon=True)
    t.start()
    worker.run()
    t.join(timeout=3)
    assert not t.is_alive()
    print("  ✓ LOAD_LAYER dispatched correctly")

    cli_net.socket.close()
    srv_net.socket.close()


def test_run_flush_cache():
    """FLUSH_CACHE with a real DiskKVCacheManager resets the cache."""
    print("\n=== run() loop: FLUSH_CACHE ===")

    tmp = tempfile.mkdtemp()
    try:
        cfg = KVCacheConfig(cache_dir=tmp, quantize_bits=4,
                            max_seq_len=16, block_size=8)
        kv_mgr = DiskKVCacheManager(cfg, n_layers=2, n_kv_heads=2, head_dim=8)
        kv_mgr.initialize()

        # Write one token so current_len > 0
        kv_mgr.save(0, 0, np.ones(16, np.float32), np.ones(16, np.float32))
        assert kv_mgr.get_current_len(0) == 1

        cli_net, srv_net = _make_network_pair()
        worker = _build_worker(cli_net, kv_mgr=kv_mgr)

        def _root():
            _send_ctrl(srv_net, OpType.FLUSH_CACHE, layer_id=0)
            srv_net.read_ack()
            _send_ctrl(srv_net, OpType.SHUTDOWN)
            srv_net.read_ack()

        t = threading.Thread(target=_root, daemon=True)
        t.start()
        worker.run()
        t.join(timeout=3)

        assert kv_mgr.get_current_len(0) == 0, \
            "KV cache must be reset after FLUSH_CACHE"
        print("  ✓ FLUSH_CACHE resets disk KV cache")

    finally:
        kv_mgr.close()
        shutil.rmtree(tmp)
        cli_net.socket.close()
        srv_net.socket.close()


def test_run_flush_cache_all_layers():
    """FLUSH_CACHE with layer_id=0xFFFFFFFF resets all layers."""
    print("\n=== run() loop: FLUSH_CACHE (all layers) ===")

    tmp = tempfile.mkdtemp()
    try:
        cfg = KVCacheConfig(cache_dir=tmp, quantize_bits=4,
                            max_seq_len=16, block_size=8)
        kv_mgr = DiskKVCacheManager(cfg, n_layers=2, n_kv_heads=2, head_dim=8)
        kv_mgr.initialize()

        vec = np.ones(16, np.float32)
        kv_mgr.save(0, 0, vec, vec)
        kv_mgr.save(1, 0, vec, vec)
        assert kv_mgr.get_current_len(0) == 1
        assert kv_mgr.get_current_len(1) == 1

        cli_net, srv_net = _make_network_pair()
        worker = _build_worker(cli_net, kv_mgr=kv_mgr)

        def _root():
            _send_ctrl(srv_net, OpType.FLUSH_CACHE, layer_id=0xFFFFFFFF)
            srv_net.read_ack()
            _send_ctrl(srv_net, OpType.SHUTDOWN)
            srv_net.read_ack()

        t = threading.Thread(target=_root, daemon=True)
        t.start()
        worker.run()
        t.join(timeout=3)

        assert kv_mgr.get_current_len(0) == 0
        assert kv_mgr.get_current_len(1) == 0
        print("  ✓ FLUSH_CACHE (all layers) resets all caches")

    finally:
        kv_mgr.close()
        shutil.rmtree(tmp)
        cli_net.socket.close()
        srv_net.socket.close()


def test_run_execute_layer():
    """EXECUTE_LAYER: root sends activations, worker returns activations."""
    print("\n=== run() loop: EXECUTE_LAYER ===")

    cli_net, srv_net = _make_network_pair()
    # Use 'none' compression so root can easily construct / parse the wire format
    worker = _build_worker(cli_net, compress='none', assigned_layers=[0])

    dim = 64
    seq_len = 1
    x_in = np.random.randn(seq_len, dim).astype(np.float32)

    def _wire_none(arr):
        data = arr.tobytes()
        return struct.pack('<IB', len(data), 0) + data   # size + method=0 + data

    received_out = []

    def _root():
        _send_ctrl(srv_net, OpType.EXECUTE_LAYER,
                   layer_id=0, offset=0, size=0, seq_len=seq_len)
        srv_net.write(_wire_none(x_in))
        # Read back: uint32 + uint8 + data
        hdr = srv_net.read(5)
        n = struct.unpack('<I', hdr[:4])[0]
        _ = hdr[4]
        out_data = srv_net.read(n)
        x_out = np.frombuffer(out_data, dtype=np.float32).reshape(seq_len, dim)
        received_out.append(x_out)
        _send_ctrl(srv_net, OpType.SHUTDOWN)
        srv_net.read_ack()

    t = threading.Thread(target=_root, daemon=True)
    t.start()
    worker.run()
    t.join(timeout=3)

    assert len(received_out) == 1
    assert np.array_equal(x_in, received_out[0]), \
        "identity engine must return x unchanged"
    print("  ✓ EXECUTE_LAYER sends and receives activations correctly")

    cli_net.socket.close()
    srv_net.socket.close()


def test_kv_cache_config_stored_on_worker():
    """KVCacheConfig passed to Worker is stored and accessible."""
    print("\n=== KVCacheConfig stored on Worker ===")

    tmp = tempfile.mkdtemp()
    try:
        cfg = KVCacheConfig(cache_dir=tmp, quantize_bits=4, max_seq_len=64)
        w = Worker('127.0.0.1', 9999, model_path=None, kv_cache_config=cfg)
        assert w.kv_cache_config is cfg, "kv_cache_config must be stored verbatim"
        print("  ✓ KVCacheConfig stored on Worker correctly")
    finally:
        shutil.rmtree(tmp)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Running worker tests...")
    print("=" * 60)

    test_network_aliases()
    test_activation_roundtrip_q80()
    test_activation_roundtrip_none()
    test_run_shutdown()
    test_run_load_layer()
    test_run_flush_cache()
    test_run_flush_cache_all_layers()
    test_run_execute_layer()
    test_kv_cache_config_stored_on_worker()

    print("=" * 60)
    print("All worker tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
