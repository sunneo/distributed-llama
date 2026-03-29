"""
Worker node implementation for Distributed-Llama in Python.

This module implements the main worker loop that:
1. Connects to the root node
2. Receives configuration (NetConfig + NodeConfig)
3. Loads model weights via memory-mapped LayerWiseInferenceEngine
4. Optionally enables disk KV cache offloading (KVCacheConfig)
5. Executes assigned transformer layers on demand
6. Synchronises activations with the root using optional Q8_0 compression

Control protocol
----------------
The root node drives the worker with ``ControlMessage`` packets (24 bytes each):

  EXECUTE_LAYER  – receive activations, run one layer, send output activations
  LOAD_LAYER     – pre-load a layer's weights into the LRU cache
  FLUSH_CACHE    – reset KV cache (start of a new sequence)
  SYNC_ACTIVATIONS – pass-through (just relay activations unchanged)
  SHUTDOWN       – clean up and exit

Activation transfer format (for EXECUTE_LAYER / SYNC_ACTIVATIONS)
------------------------------------------------------------------
Root → Worker
  1. uint32  : compressed-byte count  (little-endian)
  2. uint8   : compression method byte  (0=none, 1=q80)
  3. <n> bytes: compressed activation data

Worker → Root  (same format as above)
"""

import struct
import time
import numpy as np
from typing import Optional, Dict, List

from .network import NetworkClient
from .config import ConfigReader, NetConfig, NodeConfig
from .control_protocol import ControlMessage, OpType

# ---------------------------------------------------------------------------
# Import AirLLM components (with development fallback)
# ---------------------------------------------------------------------------
try:
    from airllm.layer_engine import LayerWiseInferenceEngine
    from airllm.kv_cache import KVCacheConfig, DiskKVCacheManager
    from airllm.activation_compression import (
        compress_activations, decompress_activations,
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from airllm.layer_engine import LayerWiseInferenceEngine
    from airllm.kv_cache import KVCacheConfig, DiskKVCacheManager
    from airllm.activation_compression import (
        compress_activations, decompress_activations,
    )

# ---------------------------------------------------------------------------
# Compression method byte codes (must match root-node convention)
# ---------------------------------------------------------------------------
_COMPRESS_NONE = 0
_COMPRESS_Q80  = 1


class Worker:
    """
    Distributed-Llama Python Worker Node.

    Connects to a C++ root node, loads assigned transformer layers
    (memory-mapped, layer-wise) and executes them on demand.
    Supports disk KV cache offloading via :class:`~airllm.kv_cache.KVCacheConfig`.
    """

    def __init__(
        self,
        host: str,
        port: int,
        model_path: Optional[str] = None,
        kv_cache_config: Optional[KVCacheConfig] = None,
        compress_activations_method: str = 'q80',
    ):
        """
        Parameters
        ----------
        host:
            Root node hostname or IP address.
        port:
            Root node TCP port.
        model_path:
            Path to the local model file (memory-mapped).
        kv_cache_config:
            If provided, KV cache is stored on disk/SSD instead of RAM.
            Pass ``None`` to keep KV cache in RAM (default).
        compress_activations_method:
            Compression for activation tensors sent over the network.
            ``'q80'`` (default) gives ~4× traffic reduction;
            ``'none'`` disables compression.
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        self.kv_cache_config = kv_cache_config
        self.compress_activations_method = compress_activations_method

        self.network: Optional[NetworkClient] = None
        self.net_config: Optional[NetConfig] = None
        self.node_config: Optional[NodeConfig] = None

        # Activation buffers keyed by pipe index
        self.pipes: Dict[int, np.ndarray] = {}

        # Layer-wise inference engine (weights kept on disk, LRU-cached in RAM)
        self.inference_engine: Optional[LayerWiseInferenceEngine] = None

        # Layers assigned to this worker node
        self.assigned_layers: List[int] = []

    # ── connection / setup ────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to the root node and receive network + node configuration."""
        print(f"Connecting to root node at {self.host}:{self.port}...")

        self.network = NetworkClient(self.host, self.port)
        self.network.connect()

        config_reader = ConfigReader(self.network)

        print("Reading network configuration...")
        self.net_config = config_reader.read_net_config()
        print(f"  Nodes: {self.net_config.n_nodes}")
        print(f"  Batches: {self.net_config.n_batches}")
        print(f"  Pipes: {self.net_config.n_pipes}")

        print("Reading node configuration...")
        self.node_config = config_reader.read_node_config()
        print(f"  Node index: {self.node_config.node_index}")
        print(f"  Segments: {self.node_config.n_segments}")
        print(f"  Buffers: {self.node_config.n_buffers}")

        self._allocate_pipes()
        print("Worker initialisation complete")

    def _allocate_pipes(self) -> None:
        """Allocate float32 activation buffers for each pipe."""
        print("Allocating activation pipes...")
        for i, pipe_cfg in enumerate(self.net_config.pipes):
            n_floats = pipe_cfg.size // 4  # F32 = 4 bytes per element
            self.pipes[i] = np.zeros(n_floats, dtype=np.float32)
            print(f"  Pipe {i} ({pipe_cfg.name}): {pipe_cfg.size} bytes")

    def load_weights(self) -> None:
        """
        Initialise the layer-wise inference engine with memory-mapped weights.

        Also sets up disk KV cache (if ``kv_cache_config`` is provided) and
        determines which transformer layers are assigned to this node.
        """
        if not self.model_path:
            print("WARNING: No model path specified, weights not loaded")
            return

        print(f"Initialising layer-wise inference engine: {self.model_path}")
        self.inference_engine = LayerWiseInferenceEngine(
            self.model_path,
            kv_cache_config=self.kv_cache_config,
        )
        self.inference_engine.initialize()

        if self.inference_engine.header:
            total_layers = self.inference_engine.header.n_layers
            n_nodes  = self.net_config.n_nodes  if self.net_config  else 1
            node_idx = self.node_config.node_index if self.node_config else 0

            # Simple round-robin layer assignment
            self.assigned_layers = list(range(node_idx, total_layers, n_nodes))
            print(f"Assigned layers: {self.assigned_layers}")

            if self.kv_cache_config is not None:
                print(
                    f"Disk KV cache: {self.kv_cache_config.cache_dir}  "
                    f"Q{self.kv_cache_config.quantize_bits}_0  "
                    f"max_seq={self.kv_cache_config.max_seq_len}"
                )

    # ── activation transfer ───────────────────────────────────────────────────

    def receive_activations(self, expected_shape: Optional[tuple] = None) -> np.ndarray:
        """
        Receive an activation tensor from the root node.

        Wire format (root → worker):
          - uint32  : compressed byte count  (4 bytes, little-endian)
          - uint8   : compression method code (1 byte)
          - <n>bytes: compressed activation data

        Parameters
        ----------
        expected_shape:
            When provided the received data is reshaped to this shape.
            When ``None`` a flat float32 array is returned.

        Returns
        -------
        np.ndarray
            Decoded float32 activation tensor.
        """
        if self.network is None:
            raise RuntimeError("Not connected to network")

        # Read header: compressed_size (uint32) + method byte
        hdr = self.network.read(5)
        compressed_size = struct.unpack('<I', hdr[:4])[0]
        method_byte     = hdr[4]

        method = {_COMPRESS_NONE: 'none', _COMPRESS_Q80: 'q80'}.get(
            method_byte, 'none'
        )

        # Read compressed payload
        data = self.network.read(compressed_size)

        # Decompress
        if expected_shape is None:
            # Fallback: treat as flat F32 (no compression assumed for unknown shape)
            activations = np.frombuffer(data, dtype=np.float32).copy()
        else:
            activations = decompress_activations(data, expected_shape, method=method)

        return activations

    def send_activations(self, activations: np.ndarray) -> None:
        """
        Send an activation tensor to the root node.

        Wire format (worker → root):
          - uint32  : compressed byte count  (4 bytes, little-endian)
          - uint8   : compression method code (1 byte)
          - <n>bytes: compressed activation data

        Parameters
        ----------
        activations:
            Float32 tensor to transmit.
        """
        if self.network is None:
            raise RuntimeError("Not connected to network")

        method = self.compress_activations_method
        data   = compress_activations(activations, method=method)

        method_byte = {
            'none': _COMPRESS_NONE,
            'q80':  _COMPRESS_Q80,
        }.get(method, _COMPRESS_NONE)

        # Send header + payload atomically
        hdr = struct.pack('<IB', len(data), method_byte)
        self.network.write(hdr + data)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Main worker loop – dispatch :class:`ControlMessage` commands from root.

        Supported operations
        --------------------
        LOAD_LAYER
            Pre-load a layer's weights into the LRU weight cache without
            executing it (useful for prefetching while the previous layer runs).

        EXECUTE_LAYER
            Receive input activations from root, execute one transformer layer
            (using the disk KV cache if configured), and send output
            activations back.  ``msg.offset`` carries the sequence position;
            ``msg.seq_len`` carries the input sequence length.

        FLUSH_CACHE
            Reset the KV cache for all (or a specific) layer so the worker
            is ready for a new sequence.  ``msg.layer_id == 0xFFFFFFFF``
            resets all layers; any other value resets that single layer.

        SYNC_ACTIVATIONS
            Receive ``msg.size`` bytes of activations from root and relay them
            unchanged to root (pass-through synchronisation).

        SHUTDOWN
            Acknowledge and exit the loop cleanly.
        """
        if self.inference_engine is None:
            raise RuntimeError("Inference engine not initialised — call load_weights() first")

        kv_mgr = self.inference_engine.kv_cache_manager  # may be None

        print("Starting worker main loop...")
        print(f"Assigned layers: {self.assigned_layers}")
        print(f"Activation compression: {self.compress_activations_method}")

        try:
            while True:
                # ── receive control message ───────────────────────────────
                raw = self.network.read(24)
                msg = ControlMessage.from_bytes(raw)

                # ── dispatch ──────────────────────────────────────────────

                if msg.op_type == OpType.SHUTDOWN:
                    print("Received SHUTDOWN signal – exiting loop")
                    self.network.write_ack()
                    break

                elif msg.op_type == OpType.FLUSH_CACHE:
                    # Reset KV cache for one layer or all layers
                    if kv_mgr is not None:
                        target = None if msg.layer_id == 0xFFFFFFFF else msg.layer_id
                        kv_mgr.reset(target)
                        print(
                            f"KV cache flushed: "
                            f"{'all layers' if target is None else f'layer {target}'}"
                        )
                    self.network.write_ack()

                elif msg.op_type == OpType.LOAD_LAYER:
                    # Pre-load weights (no execution)
                    layer_id = msg.layer_id
                    if layer_id in self.assigned_layers:
                        self.inference_engine.load_layer(layer_id, prefetch_next=False)
                        print(f"Pre-loaded layer {layer_id}")
                    self.network.write_ack()

                elif msg.op_type == OpType.EXECUTE_LAYER:
                    layer_id = msg.layer_id
                    pos      = int(msg.offset)
                    seq_len  = int(msg.seq_len) if msg.seq_len > 0 else 1

                    # Determine expected activation shape from model header
                    if self.inference_engine.header is not None:
                        dim = self.inference_engine.header.dim
                        act_shape = (seq_len, dim)
                    else:
                        act_shape = None

                    # Receive input activations from root
                    x = self.receive_activations(expected_shape=act_shape)
                    if act_shape is not None:
                        x = x.reshape(act_shape)

                    # Execute the layer (with disk KV cache if configured)
                    if layer_id in self.assigned_layers:
                        x, _ = self.inference_engine.execute_layer(
                            layer_id, x, pos,
                            kv_cache_manager=kv_mgr,
                        )
                    # else: this worker isn't responsible – relay unchanged

                    # Send output activations back to root
                    self.send_activations(x)

                elif msg.op_type == OpType.SYNC_ACTIVATIONS:
                    # Pure relay: receive and retransmit unchanged
                    size = int(msg.size)
                    if size > 0:
                        raw_act = self.network.read(size)
                        self.network.write(raw_act)
                    else:
                        self.network.write_ack()

                else:
                    print(f"WARNING: Unknown op_type {msg.op_type} – skipping")
                    self.network.write_ack()

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt – shutting down worker")
        except Exception as exc:
            print(f"Error in worker loop: {exc}")
            raise

    # ── cleanup ───────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean up resources and disconnect from root."""
        if self.inference_engine:
            self.inference_engine.cleanup()
        if self.network:
            self.network.disconnect()
        print("Worker shutdown complete")


# ── command-line entry point ──────────────────────────────────────────────────

def main():
    """CLI entry point for the Python worker node."""
    import argparse

    parser = argparse.ArgumentParser(description='Distributed-Llama Python Worker')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Root node hostname/IP (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9999,
                        help='Root node port (default: 9999)')
    parser.add_argument('--model', default=None,
                        help='Path to model file')
    parser.add_argument('--compress', default='q80',
                        choices=['none', 'q80'],
                        help='Activation compression for network transfer (default: q80)')

    # KV cache disk offloading arguments
    kv_group = parser.add_argument_group('KV cache disk offloading')
    kv_group.add_argument(
        '--kv-cache-dir', default=None,
        help='Directory for KV cache mmap files.  '
             'When set, KV cache is offloaded to disk/SSD instead of RAM.',
    )
    kv_group.add_argument(
        '--kv-cache-bits', type=int, default=4, choices=[4, 8],
        help='KV cache quantisation bits: 4 (Q4_0, ~87%% I/O savings, default) '
             'or 8 (Q8_0, ~75%% I/O savings).',
    )
    kv_group.add_argument(
        '--kv-cache-max-seq-len', type=int, default=2048,
        help='Maximum sequence length for disk KV cache (default: 2048)',
    )

    args = parser.parse_args()

    # Build KV cache config if a directory was supplied
    kv_cfg: Optional[KVCacheConfig] = None
    if args.kv_cache_dir:
        kv_cfg = KVCacheConfig(
            cache_dir=args.kv_cache_dir,
            quantize_bits=args.kv_cache_bits,
            max_seq_len=args.kv_cache_max_seq_len,
        )
        print(
            f"KV cache disk offloading: dir={args.kv_cache_dir}  "
            f"bits={args.kv_cache_bits}  max_seq={args.kv_cache_max_seq_len}"
        )

    worker = Worker(
        host=args.host,
        port=args.port,
        model_path=args.model,
        kv_cache_config=kv_cfg,
        compress_activations_method=args.compress,
    )

    try:
        worker.connect()
        worker.load_weights()
        worker.run()
    finally:
        worker.shutdown()


if __name__ == '__main__':
    main()
