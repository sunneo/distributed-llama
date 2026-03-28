"""
KV Cache Offloading for AirLLM – disk/SSD-backed KV cache with quantization.

Implements three ideas for running large models with limited RAM:

1. **Disk offloading** – KV cache entries are stored in pre-allocated
   mmap files on an SSD instead of in RAM.  The operating system handles
   the RAM↔SSD scheduling automatically (the same technique already used
   for model weights in MemoryMappedWeights).

2. **4-bit quantisation** – Key and Value tensors are compressed to Q4_0
   format before writing to disk, reducing SSD I/O by ~75–87 % compared
   with F32 storage (inspired by LMCache's approach of offloading
   quantised KV caches).

3. **Sequential append / range load** – new tokens are appended at the
   next free slot; all previous positions are loaded in a single contiguous
   read, keeping I/O patterns sequential-friendly for SSDs.

Usage
-----
    cfg = KVCacheConfig(cache_dir='/tmp/kv_cache', quantize_bits=4,
                        max_seq_len=2048)
    mgr = DiskKVCacheManager(cfg, n_layers=32, n_kv_heads=8, head_dim=128)
    mgr.initialize()

    # During generation:
    mgr.save(layer_id=0, pos=pos, k=k_vec, v=v_vec)
    k_cache, v_cache = mgr.load(layer_id=0, seq_len=pos)  # positions [0, pos)

    # Start a new sequence:
    mgr.reset()
"""

import os
import struct
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from .activation_compression import (
    quantize_f32_to_q40, dequantize_q40_to_f32,
    quantize_f32_to_q80, dequantize_q80_to_f32,
)

# ──────────────────────────────────────────────────────────────────────────────
# File-format constants
# ──────────────────────────────────────────────────────────────────────────────

_MAGIC   = 0x4B564300   # "KVC\0"
_VERSION = 1
_HEADER_BYTES = 32      # 8 × uint32

# Header field indices
_HDR_MAGIC         = 0
_HDR_VERSION       = 1
_HDR_N_KV_HEADS    = 2
_HDR_HEAD_DIM      = 3
_HDR_MAX_SEQ_LEN   = 4
_HDR_QUANTIZE_BITS = 5
_HDR_CURRENT_LEN   = 6  # mutable – updated on every save()
_HDR_BLOCK_SIZE    = 7


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KVCacheConfig:
    """Configuration for KV cache disk offloading.

    Attributes
    ----------
    cache_dir:
        Directory for mmap cache files.  Created automatically.
    quantize_bits:
        4 → Q4_0 (~87 % I/O reduction vs F32, recommended for SSDs).
        8 → Q8_0 (~75 % I/O reduction vs F32).
    max_seq_len:
        Maximum sequence length (pre-allocated in the mmap file).
    block_size:
        Quantisation block size (must be even; default 32 matches the
        existing Q8_0 block size used by activation_compression).
    """
    cache_dir: str = "/tmp/kv_cache"
    quantize_bits: int = 4
    max_seq_len: int = 2048
    block_size: int = 32


# ──────────────────────────────────────────────────────────────────────────────
# DiskKVCacheManager
# ──────────────────────────────────────────────────────────────────────────────

class DiskKVCacheManager:
    """
    Per-layer KV cache manager backed by mmap files on disk/SSD.

    Each layer gets its own binary file.  The file is pre-allocated to hold
    ``max_seq_len`` quantised K/V vectors so that all writes are in-place
    (no file growth, SSD-friendly sequential patterns).

    The OS-level page cache transparently keeps hot pages in RAM, so
    recently accessed positions are served from memory without extra copying.

    File layout (byte offsets)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    [0,   32)  Header – 8 × uint32 fields (see _HDR_* constants)
    [32,  32 + k_scales_bytes)  K scales – (max_seq_len, n_blocks) float16
    [32 + k_scales_bytes,
     32 + k_scales_bytes + k_packed_bytes)  K packed nibbles / int8
    … then identical V scales and V packed regions
    """

    def __init__(
        self,
        config: KVCacheConfig,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> None:
        """
        Parameters
        ----------
        config:
            KV cache configuration.
        n_layers:
            Total number of transformer layers (one mmap file per layer).
        n_kv_heads:
            Number of KV heads in the model.
        head_dim:
            Dimension of each KV head (``kv_dim = n_kv_heads * head_dim``).
        """
        self.config    = config
        self.n_layers  = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim  = head_dim
        self.kv_dim    = n_kv_heads * head_dim

        bs = config.block_size
        # Number of quantisation blocks per KV vector
        self._n_blocks: int = (self.kv_dim + bs - 1) // bs
        # Padded kv_dim (multiple of block_size)
        self._padded_kv_dim: int = self._n_blocks * bs

        # Byte sizes for one position's worth of data
        if config.quantize_bits == 4:
            self._values_per_pos: int = self._n_blocks * (bs // 2)  # packed nibbles
        else:  # 8-bit
            self._values_per_pos = self._padded_kv_dim  # int8

        self._scales_per_pos: int = self._n_blocks  # float16 elements

        # Open mmap handles: layer_id -> np.memmap
        self._mmaps: Dict[int, np.memmap] = {}

        # Cached byte offsets within the file (computed once)
        self._k_scales_offset: int = _HEADER_BYTES
        self._k_values_offset: int = (
            self._k_scales_offset
            + config.max_seq_len * self._scales_per_pos * 2  # float16 = 2 bytes
        )
        self._v_scales_offset: int = (
            self._k_values_offset
            + config.max_seq_len * self._values_per_pos  # uint8 = 1 byte
        )
        self._v_values_offset: int = (
            self._v_scales_offset
            + config.max_seq_len * self._scales_per_pos * 2
        )
        self._total_bytes: int = (
            self._v_values_offset
            + config.max_seq_len * self._values_per_pos
        )

    # ── public API ────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Pre-allocate mmap files for all layers.

        Safe to call multiple times – existing files are not overwritten.
        """
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        for layer_id in range(self.n_layers):
            path = self._cache_path(layer_id)
            if not path.exists():
                self._create_cache_file(layer_id, path)

    def save(self, layer_id: int, pos: int, k: np.ndarray, v: np.ndarray) -> None:
        """Quantise and persist K/V vectors for position *pos* in *layer_id*.

        Parameters
        ----------
        layer_id:
            Layer index.
        pos:
            Token position (0-based).  Must be < ``max_seq_len``.
        k:
            Key vector, shape ``(kv_dim,)`` or ``(1, kv_dim)``, dtype float32.
        v:
            Value vector, same shape as *k*.
        """
        if pos >= self.config.max_seq_len:
            raise IndexError(
                f"pos={pos} exceeds max_seq_len={self.config.max_seq_len}"
            )

        k_flat = k.flatten().astype(np.float32)
        v_flat = v.flatten().astype(np.float32)

        mm = self._get_mmap(layer_id)

        # Quantise and write K
        k_scales, k_packed = self._quantize(k_flat)
        self._write_scales(mm, self._k_scales_offset, pos, k_scales)
        self._write_values(mm, self._k_values_offset, pos, k_packed)

        # Quantise and write V
        v_scales, v_packed = self._quantize(v_flat)
        self._write_scales(mm, self._v_scales_offset, pos, v_scales)
        self._write_values(mm, self._v_values_offset, pos, v_packed)

        # Update current_len in header (advancing only forward)
        hdr = self._read_header(mm)
        new_len = max(int(hdr[_HDR_CURRENT_LEN]), pos + 1)
        hdr[_HDR_CURRENT_LEN] = new_len
        mm[:_HEADER_BYTES] = hdr.view(np.uint8)

    def load(
        self, layer_id: int, seq_len: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load accumulated KV cache for positions ``[0, seq_len)``.

        Parameters
        ----------
        layer_id:
            Layer index.
        seq_len:
            Number of positions to load (exclusive upper bound).

        Returns
        -------
        ``(k_cache, v_cache)`` each of shape ``(seq_len, kv_dim)`` in F32,
        or ``None`` if *seq_len* is 0 or the file does not exist.
        """
        if seq_len <= 0:
            return None

        path = self._cache_path(layer_id)
        if not path.exists():
            return None

        mm = self._get_mmap(layer_id)
        current_len = int(self._read_header(mm)[_HDR_CURRENT_LEN])
        seq_len = min(seq_len, current_len)
        if seq_len <= 0:
            return None

        k = self._load_kv(mm, self._k_scales_offset, self._k_values_offset, seq_len)
        v = self._load_kv(mm, self._v_scales_offset, self._v_values_offset, seq_len)
        return k, v

    def get_current_len(self, layer_id: int) -> int:
        """Return the number of positions written for *layer_id*."""
        path = self._cache_path(layer_id)
        if not path.exists():
            return 0
        mm = self._get_mmap(layer_id)
        return int(self._read_header(mm)[_HDR_CURRENT_LEN])

    def reset(self, layer_id: Optional[int] = None) -> None:
        """Reset the current-length counter to 0 for one or all layers.

        Parameters
        ----------
        layer_id:
            If given, reset only that layer's cache.  If ``None``, reset
            all layers.
        """
        layer_ids = [layer_id] if layer_id is not None else range(self.n_layers)
        for lid in layer_ids:
            path = self._cache_path(lid)
            if not path.exists():
                continue
            mm = self._get_mmap(lid)
            hdr = self._read_header(mm)
            hdr[_HDR_CURRENT_LEN] = 0
            mm[:_HEADER_BYTES] = hdr.view(np.uint8)

    def close(self) -> None:
        """Flush and close all open mmap handles."""
        for mm in self._mmaps.values():
            mm.flush()
            del mm
        self._mmaps.clear()

    def get_stats(self) -> dict:
        """Return a summary of the cache configuration and size estimates."""
        cfg = self.config
        bs  = cfg.block_size

        k_scales_bytes = cfg.max_seq_len * self._scales_per_pos * 2
        k_values_bytes = cfg.max_seq_len * self._values_per_pos
        per_layer_bytes = _HEADER_BYTES + 2 * (k_scales_bytes + k_values_bytes)
        total_bytes = per_layer_bytes * self.n_layers

        f32_bytes = cfg.max_seq_len * self.kv_dim * 4 * 2  # K + V, F32
        return {
            'n_layers': self.n_layers,
            'n_kv_heads': self.n_kv_heads,
            'head_dim': self.head_dim,
            'kv_dim': self.kv_dim,
            'max_seq_len': cfg.max_seq_len,
            'quantize_bits': cfg.quantize_bits,
            'block_size': bs,
            'per_layer_bytes': per_layer_bytes,
            'total_bytes_all_layers': total_bytes,
            'total_mb_all_layers': total_bytes / (1024 ** 2),
            'f32_bytes_all_layers': f32_bytes * self.n_layers,
            'savings_percent': (1.0 - total_bytes / (f32_bytes * self.n_layers)) * 100
            if f32_bytes > 0 else 0.0,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _cache_path(self, layer_id: int) -> Path:
        return Path(self.config.cache_dir) / f"kv_layer_{layer_id:04d}.bin"

    def _create_cache_file(self, layer_id: int, path: Path) -> None:
        """Create a zero-filled, pre-allocated mmap file with header."""
        mm = np.memmap(str(path), dtype=np.uint8, mode='w+',
                       shape=(self._total_bytes,))

        hdr = np.zeros(8, dtype=np.uint32)
        hdr[_HDR_MAGIC]         = _MAGIC
        hdr[_HDR_VERSION]       = _VERSION
        hdr[_HDR_N_KV_HEADS]    = self.n_kv_heads
        hdr[_HDR_HEAD_DIM]      = self.head_dim
        hdr[_HDR_MAX_SEQ_LEN]   = self.config.max_seq_len
        hdr[_HDR_QUANTIZE_BITS] = self.config.quantize_bits
        hdr[_HDR_CURRENT_LEN]   = 0
        hdr[_HDR_BLOCK_SIZE]    = self.config.block_size

        mm[:_HEADER_BYTES] = hdr.view(np.uint8)
        mm.flush()
        del mm

    def _get_mmap(self, layer_id: int) -> np.memmap:
        """Return (and cache) the mmap handle for *layer_id*."""
        if layer_id not in self._mmaps:
            path = self._cache_path(layer_id)
            if not path.exists():
                self._create_cache_file(layer_id, path)
            self._mmaps[layer_id] = np.memmap(
                str(path), dtype=np.uint8, mode='r+',
                shape=(self._total_bytes,)
            )
        return self._mmaps[layer_id]

    def _read_header(self, mm: np.memmap) -> np.ndarray:
        """Return header as a mutable uint32 array (copy)."""
        return np.frombuffer(mm[:_HEADER_BYTES].tobytes(), dtype=np.uint32).copy()

    # ── quantise / dequantise wrappers ────────────────────────────────────────

    def _quantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scales, packed_values) using the configured bit-width."""
        if self.config.quantize_bits == 4:
            return quantize_f32_to_q40(x, self.config.block_size)
        else:
            scales, quantized = quantize_f32_to_q80(x, self.config.block_size)
            return scales, quantized.view(np.uint8)

    def _dequantize(self, scales: np.ndarray, packed: np.ndarray,
                    shape: Tuple[int, ...]) -> np.ndarray:
        if self.config.quantize_bits == 4:
            return dequantize_q40_to_f32(scales, packed, shape,
                                         self.config.block_size)
        else:
            return dequantize_q80_to_f32(scales, packed.view(np.int8),
                                         shape, self.config.block_size)

    # ── mmap read / write helpers ─────────────────────────────────────────────

    def _write_scales(self, mm: np.memmap, region_offset: int,
                      pos: int, scales: np.ndarray) -> None:
        """Write scale values for a single position into the scales region."""
        byte_offset = region_offset + pos * self._scales_per_pos * 2
        mm[byte_offset: byte_offset + scales.nbytes] = scales.view(np.uint8)

    def _write_values(self, mm: np.memmap, region_offset: int,
                      pos: int, packed: np.ndarray) -> None:
        """Write packed quantised values for a single position."""
        byte_offset = region_offset + pos * self._values_per_pos
        mm[byte_offset: byte_offset + packed.nbytes] = packed.view(np.uint8)

    def _load_kv(self, mm: np.memmap, scales_offset: int,
                 values_offset: int, seq_len: int) -> np.ndarray:
        """Load and dequantise seq_len vectors from one K or V region."""
        # Read scales: shape (seq_len, n_blocks), dtype float16
        s_bytes = seq_len * self._scales_per_pos * 2
        scales_raw = np.frombuffer(
            mm[scales_offset: scales_offset + s_bytes].tobytes(),
            dtype=np.float16
        ).reshape(seq_len, self._scales_per_pos)

        # Read packed values: shape (seq_len, values_per_pos), dtype uint8
        v_bytes = seq_len * self._values_per_pos
        values_raw = np.frombuffer(
            mm[values_offset: values_offset + v_bytes].tobytes(),
            dtype=np.uint8
        ).reshape(seq_len, self._values_per_pos)

        # Dequantise each position
        out = np.empty((seq_len, self.kv_dim), dtype=np.float32)
        for i in range(seq_len):
            out[i] = self._dequantize(
                scales_raw[i],
                values_raw[i],
                (self.kv_dim,)
            )
        return out
