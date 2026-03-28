"""
Tests for KV Cache disk offloading (kv_cache.py + Q4_0 quantisation).

Tests cover:
- Q4_0 quantise / dequantise round-trip
- Q4_0 pack / unpack binary storage format
- compress_activations / decompress_activations with 'q40' method
- calculate_compression_ratio for 'q40'
- DiskKVCacheManager: file creation, save, load, reset
- DiskKVCacheManager: Q8_0 mode
- Integration: multiple layers, sequential token generation
"""

import sys
import os
import tempfile
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# ── helper ────────────────────────────────────────────────────────────────────

def allclose_q4(a, b, atol: float = 0.5) -> bool:
    """Q4_0 has a coarser grid than Q8_0; allow larger absolute tolerance."""
    return bool(np.allclose(a, b, atol=atol))


# ── Q4_0 quantisation ─────────────────────────────────────────────────────────

def test_q40_quantize_dequantize():
    """Round-trip: F32 → Q4_0 → F32."""
    print("\n=== Q4_0 quantise / dequantise round-trip ===")

    from airllm.activation_compression import quantize_f32_to_q40, dequantize_q40_to_f32

    shape = (4, 128)
    x = np.random.randn(*shape).astype(np.float32)

    scales, packed = quantize_f32_to_q40(x)

    assert scales.dtype == np.float16, "scales must be float16"
    assert packed.dtype == np.uint8,   "packed must be uint8"

    n_elements = np.prod(shape)
    n_blocks   = (n_elements + 32 - 1) // 32
    assert len(scales) == n_blocks,       f"expected {n_blocks} scales, got {len(scales)}"
    assert len(packed) == n_blocks * 16,  f"expected {n_blocks * 16} packed bytes"

    x_restored = dequantize_q40_to_f32(scales, packed, shape)
    assert x_restored.shape == shape, "shape mismatch after dequantisation"

    mse = float(np.mean((x - x_restored) ** 2))
    max_err = float(np.max(np.abs(x - x_restored)))
    print(f"  MSE={mse:.4f}  max_err={max_err:.4f}")
    assert mse < 0.5, f"MSE too large: {mse}"
    assert max_err < 3.0, f"max_error too large: {max_err}"

    print("  ✓ Q4_0 round-trip passed")


def test_q40_zero_tensor():
    """All-zeros tensor should survive quantisation unchanged."""
    print("\n=== Q4_0 zero tensor ===")
    from airllm.activation_compression import quantize_f32_to_q40, dequantize_q40_to_f32

    x = np.zeros((2, 64), dtype=np.float32)
    scales, packed = quantize_f32_to_q40(x)
    x_restored = dequantize_q40_to_f32(scales, packed, x.shape)

    assert np.allclose(x_restored, 0.0), "zero tensor should restore to zeros"
    print("  ✓ zero tensor passed")


def test_q40_non_multiple_of_block():
    """Tensor size not a multiple of block_size should be handled via padding."""
    print("\n=== Q4_0 non-multiple-of-block-size ===")
    from airllm.activation_compression import quantize_f32_to_q40, dequantize_q40_to_f32

    shape = (3, 50)  # 150 elements – not a multiple of 32
    x = np.random.randn(*shape).astype(np.float32)
    scales, packed = quantize_f32_to_q40(x, block_size=32)
    x_restored = dequantize_q40_to_f32(scales, packed, shape, block_size=32)
    assert x_restored.shape == shape, "shape must be preserved"
    assert allclose_q4(x, x_restored), "restored values differ too much"
    print("  ✓ non-multiple-of-block-size passed")


def test_q40_pack_unpack():
    """Binary storage format round-trip."""
    print("\n=== Q4_0 pack / unpack ===")
    from airllm.activation_compression import (
        quantize_f32_to_q40, dequantize_q40_to_f32,
        pack_q40_for_storage, unpack_q40_from_storage,
    )

    shape = (1, 256)
    x = np.random.randn(*shape).astype(np.float32)

    scales, packed = quantize_f32_to_q40(x)
    blob = pack_q40_for_storage(scales, packed)

    scales2, packed2 = unpack_q40_from_storage(blob)
    assert len(scales2) == len(scales), "scale count mismatch after unpack"
    assert len(packed2) == len(packed), "packed length mismatch after unpack"
    assert np.allclose(scales.astype(np.float32), scales2.astype(np.float32)), \
        "scales differ after pack/unpack"
    assert np.array_equal(packed, packed2), "packed values differ after pack/unpack"

    x_restored = dequantize_q40_to_f32(scales2, packed2, shape)
    assert allclose_q4(x, x_restored)
    print("  ✓ Q4_0 pack/unpack passed")


def test_compress_decompress_q40():
    """compress_activations / decompress_activations with method='q40'."""
    print("\n=== compress/decompress q40 ===")
    from airllm.activation_compression import compress_activations, decompress_activations

    shape = (1, 4096)
    x = np.random.randn(*shape).astype(np.float32)

    blob = compress_activations(x, method='q40')
    x_restored = decompress_activations(blob, shape, method='q40')

    assert x_restored.shape == shape
    # Q4_0 gives ~7x compression ratio; compressed size must be < 50% of F32
    compressed_ratio = len(blob) / x.nbytes
    print(f"  compression ratio (blob/F32): {compressed_ratio:.3f}")
    assert compressed_ratio < 0.50, f"compression ratio too high: {compressed_ratio:.3f}"
    print("  ✓ compress/decompress q40 passed")


def test_compression_ratio_q40():
    """calculate_compression_ratio should report >75% savings for q40."""
    print("\n=== compression ratio q40 ===")
    from airllm.activation_compression import calculate_compression_ratio

    stats = calculate_compression_ratio((1, 4096), method='q40')
    print(f"  savings: {stats['savings_percent']:.1f}%  ratio: {stats['compression_ratio']:.2f}x")
    assert stats['savings_percent'] >= 75.0, \
        f"expected >=75% savings, got {stats['savings_percent']:.1f}%"
    print("  ✓ compression ratio q40 passed")


# ── DiskKVCacheManager ────────────────────────────────────────────────────────

def _make_manager(tmp_dir, quantize_bits=4, max_seq_len=16,
                  n_layers=2, n_kv_heads=2, head_dim=8):
    from airllm.kv_cache import KVCacheConfig, DiskKVCacheManager
    cfg = KVCacheConfig(
        cache_dir=tmp_dir,
        quantize_bits=quantize_bits,
        max_seq_len=max_seq_len,
        block_size=8,  # small block size for faster tests
    )
    mgr = DiskKVCacheManager(cfg, n_layers, n_kv_heads, head_dim)
    mgr.initialize()
    return mgr


def test_disk_kv_cache_init():
    """initialize() creates one mmap file per layer."""
    print("\n=== DiskKVCacheManager init ===")
    tmp = tempfile.mkdtemp()
    try:
        mgr = _make_manager(tmp, n_layers=3)
        import glob as glob_mod
        files = glob_mod.glob(os.path.join(tmp, "kv_layer_*.bin"))
        assert len(files) == 3, f"expected 3 cache files, got {len(files)}"
        print(f"  created {len(files)} cache files")
        print("  ✓ init passed")
    finally:
        mgr.close()
        shutil.rmtree(tmp)


def test_disk_kv_cache_save_load_q40():
    """save() then load() round-trip with Q4_0."""
    print("\n=== DiskKVCacheManager save/load Q4_0 ===")
    tmp = tempfile.mkdtemp()
    try:
        n_kv_heads, head_dim = 2, 8
        kv_dim = n_kv_heads * head_dim
        mgr = _make_manager(tmp, quantize_bits=4, n_kv_heads=n_kv_heads,
                            head_dim=head_dim, max_seq_len=16)

        # Write 4 positions
        k_vecs, v_vecs = [], []
        for pos in range(4):
            k = np.random.randn(kv_dim).astype(np.float32)
            v = np.random.randn(kv_dim).astype(np.float32)
            k_vecs.append(k)
            v_vecs.append(v)
            mgr.save(layer_id=0, pos=pos, k=k, v=v)

        # Check current_len
        assert mgr.get_current_len(0) == 4, "current_len should be 4"

        # Load positions [0, 4)
        result = mgr.load(layer_id=0, seq_len=4)
        assert result is not None, "load should return tensors"
        k_loaded, v_loaded = result
        assert k_loaded.shape == (4, kv_dim), f"K shape mismatch: {k_loaded.shape}"
        assert v_loaded.shape == (4, kv_dim), f"V shape mismatch: {v_loaded.shape}"

        # Compare each position (Q4_0 tolerance)
        k_orig = np.stack(k_vecs)
        v_orig = np.stack(v_vecs)
        k_mse  = float(np.mean((k_orig - k_loaded) ** 2))
        v_mse  = float(np.mean((v_orig - v_loaded) ** 2))
        print(f"  K MSE={k_mse:.4f}  V MSE={v_mse:.4f}")
        assert k_mse < 0.5, f"K MSE too large: {k_mse}"
        assert v_mse < 0.5, f"V MSE too large: {v_mse}"

        print("  ✓ save/load Q4_0 passed")
    finally:
        mgr.close()
        shutil.rmtree(tmp)


def test_disk_kv_cache_save_load_q80():
    """save() then load() round-trip with Q8_0."""
    print("\n=== DiskKVCacheManager save/load Q8_0 ===")
    tmp = tempfile.mkdtemp()
    try:
        n_kv_heads, head_dim = 2, 8
        kv_dim = n_kv_heads * head_dim
        mgr = _make_manager(tmp, quantize_bits=8, n_kv_heads=n_kv_heads,
                            head_dim=head_dim, max_seq_len=16)

        k = np.random.randn(kv_dim).astype(np.float32)
        v = np.random.randn(kv_dim).astype(np.float32)
        mgr.save(layer_id=0, pos=0, k=k, v=v)

        result = mgr.load(layer_id=0, seq_len=1)
        assert result is not None
        k_loaded, v_loaded = result

        k_mse = float(np.mean((k - k_loaded[0]) ** 2))
        v_mse = float(np.mean((v - v_loaded[0]) ** 2))
        print(f"  K MSE={k_mse:.6f}  V MSE={v_mse:.6f}")
        assert k_mse < 0.01, f"Q8_0 K MSE too large: {k_mse}"
        assert v_mse < 0.01, f"Q8_0 V MSE too large: {v_mse}"

        print("  ✓ save/load Q8_0 passed")
    finally:
        mgr.close()
        shutil.rmtree(tmp)


def test_disk_kv_cache_load_empty():
    """load() with seq_len=0 should return None."""
    print("\n=== DiskKVCacheManager load empty ===")
    tmp = tempfile.mkdtemp()
    try:
        mgr = _make_manager(tmp)
        result = mgr.load(layer_id=0, seq_len=0)
        assert result is None, "expected None for seq_len=0"
        print("  ✓ load empty passed")
    finally:
        mgr.close()
        shutil.rmtree(tmp)


def test_disk_kv_cache_reset():
    """reset() clears current_len so the cache can be reused."""
    print("\n=== DiskKVCacheManager reset ===")
    tmp = tempfile.mkdtemp()
    try:
        n_kv_heads, head_dim = 2, 8
        kv_dim = n_kv_heads * head_dim
        mgr = _make_manager(tmp, n_kv_heads=n_kv_heads, head_dim=head_dim)

        k = np.random.randn(kv_dim).astype(np.float32)
        v = np.random.randn(kv_dim).astype(np.float32)
        for pos in range(5):
            mgr.save(layer_id=0, pos=pos, k=k, v=v)
        assert mgr.get_current_len(0) == 5

        mgr.reset(layer_id=0)
        assert mgr.get_current_len(0) == 0, "current_len should be 0 after reset"

        # load after reset should return None
        result = mgr.load(layer_id=0, seq_len=1)
        assert result is None, "should return None after reset"
        print("  ✓ reset passed")
    finally:
        mgr.close()
        shutil.rmtree(tmp)


def test_disk_kv_cache_multiple_layers():
    """Each layer's cache is independent."""
    print("\n=== DiskKVCacheManager multiple layers ===")
    tmp = tempfile.mkdtemp()
    try:
        n_kv_heads, head_dim = 2, 8
        kv_dim = n_kv_heads * head_dim
        n_layers = 4
        mgr = _make_manager(tmp, n_kv_heads=n_kv_heads, head_dim=head_dim,
                            n_layers=n_layers, max_seq_len=16)

        # Write different data to each layer
        k_data = {lid: np.random.randn(kv_dim).astype(np.float32)
                  for lid in range(n_layers)}

        for lid in range(n_layers):
            mgr.save(layer_id=lid, pos=0, k=k_data[lid], v=k_data[lid])

        for lid in range(n_layers):
            result = mgr.load(layer_id=lid, seq_len=1)
            assert result is not None
            k_loaded, _ = result
            mse = float(np.mean((k_data[lid] - k_loaded[0]) ** 2))
            assert mse < 0.5, f"layer {lid}: MSE too large: {mse}"

        print(f"  ✓ {n_layers} independent layer caches passed")
    finally:
        mgr.close()
        shutil.rmtree(tmp)


def test_disk_kv_cache_stats():
    """get_stats() reports expected savings percentage."""
    print("\n=== DiskKVCacheManager get_stats ===")
    tmp = tempfile.mkdtemp()
    try:
        mgr = _make_manager(tmp, quantize_bits=4, n_kv_heads=2, head_dim=8)
        stats = mgr.get_stats()
        print(f"  savings: {stats['savings_percent']:.1f}%")
        assert stats['savings_percent'] > 0, "should have positive savings"
        assert stats['n_layers'] == 2
        assert stats['kv_dim'] == 16
        print("  ✓ get_stats passed")
    finally:
        mgr.close()
        shutil.rmtree(tmp)


# ── integration ───────────────────────────────────────────────────────────────

def test_integration_sequential_generation():
    """
    Simulate a short autoregressive generation loop using DiskKVCacheManager,
    verifying that the accumulated KV cache grows correctly.
    """
    print("\n=== Integration: sequential token generation ===")
    tmp = tempfile.mkdtemp()
    try:
        n_kv_heads, head_dim = 2, 8
        kv_dim  = n_kv_heads * head_dim
        n_layers = 2
        seq_len_total = 6

        from airllm.kv_cache import KVCacheConfig, DiskKVCacheManager
        cfg = KVCacheConfig(cache_dir=tmp, quantize_bits=4,
                            max_seq_len=32, block_size=8)
        mgr = DiskKVCacheManager(cfg, n_layers, n_kv_heads, head_dim)
        mgr.initialize()

        # Simulate generating seq_len_total tokens one by one
        for pos in range(seq_len_total):
            for lid in range(n_layers):
                k_new = np.random.randn(kv_dim).astype(np.float32)
                v_new = np.random.randn(kv_dim).astype(np.float32)

                # Load accumulated cache for attention
                past = mgr.load(lid, pos)
                if past is not None:
                    k_cache, v_cache = past
                    assert k_cache.shape == (pos, kv_dim), \
                        f"pos={pos} lid={lid}: k_cache shape {k_cache.shape}"

                # Save new token's K/V
                mgr.save(lid, pos, k_new, v_new)

        # Verify final lengths
        for lid in range(n_layers):
            assert mgr.get_current_len(lid) == seq_len_total, \
                f"layer {lid}: expected len {seq_len_total}"

        print(f"  Generated {seq_len_total} tokens × {n_layers} layers successfully")
        print("  ✓ integration test passed")

    finally:
        mgr.close()
        shutil.rmtree(tmp)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Running KV cache offloading tests...")
    print("=" * 60)

    test_q40_quantize_dequantize()
    test_q40_zero_tensor()
    test_q40_non_multiple_of_block()
    test_q40_pack_unpack()
    test_compress_decompress_q40()
    test_compression_ratio_q40()

    test_disk_kv_cache_init()
    test_disk_kv_cache_save_load_q40()
    test_disk_kv_cache_save_load_q80()
    test_disk_kv_cache_load_empty()
    test_disk_kv_cache_reset()
    test_disk_kv_cache_multiple_layers()
    test_disk_kv_cache_stats()

    test_integration_sequential_generation()

    print("=" * 60)
    print("All KV cache tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
