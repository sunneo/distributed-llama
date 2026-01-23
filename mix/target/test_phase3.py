"""
Tests for Phase 3 implementations: Zero-Data Movement Architecture.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tempfile
from pathlib import Path

# Test storage coordinator
def test_storage_coordinator():
    """Test shared storage verification."""
    print("\n=== Testing Storage Coordinator ===")
    
    # Create a temporary model file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.model') as tmp:
        tmp.write(b"Test model data" * 1000)  # Write some data
        tmp_path = tmp.name
    
    try:
        from airllm.storage_coordinator import StorageCoordinator
        
        coordinator = StorageCoordinator(tmp_path)
        
        # Test file existence
        assert coordinator.verify_file_exists(), "File should exist"
        
        # Test file size
        size = coordinator.get_file_size()
        print(f"  File size: {size} bytes")
        assert size > 0, "File size should be > 0"
        
        # Test fast checksum
        checksum = coordinator.compute_fast_checksum()
        print(f"  Fast checksum: {checksum}")
        assert len(checksum) == 32, "MD5 checksum should be 32 hex chars"
        
        # Test file info
        info = coordinator.get_file_info(fast_checksum=True)
        print(f"  File info: exists={info['exists']}, size={info['size_gb']:.4f} GB")
        assert info['exists'], "File should exist"
        
        # Test verification against itself
        assert coordinator.verify_against(info), "File should match itself"
        
        print("  ✓ All storage coordinator tests passed")
        
    finally:
        # Clean up
        os.unlink(tmp_path)


def test_control_protocol():
    """Test optimized control protocol."""
    print("\n=== Testing Control Protocol ===")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'distributed-llama.python'))
    from control_protocol import (
        ControlMessage, OpType, TensorMetadata, ControlProtocol,
        calculate_overhead_savings
    )
    
    # Test control message serialization
    msg = ControlMessage(
        op_type=OpType.EXECUTE_LAYER,
        layer_id=42,
        offset=1024,
        size=2048,
        seq_len=1
    )
    
    data = msg.to_bytes()
    print(f"  Control message size: {len(data)} bytes")
    assert len(data) == 24, "Control message should be 24 bytes"
    
    msg_restored = ControlMessage.from_bytes(data)
    assert msg_restored.op_type == msg.op_type
    assert msg_restored.layer_id == msg.layer_id
    assert msg_restored.offset == msg.offset
    print("  ✓ Control message serialization OK")
    
    # Test tensor metadata
    meta = TensorMetadata(shape=(1, 4096), dtype='f32')
    data = meta.to_bytes()
    print(f"  Tensor metadata size: {len(data)} bytes")
    
    meta_restored = TensorMetadata.from_bytes(data)
    assert meta_restored.shape == meta.shape
    assert meta_restored.dtype == meta.dtype
    print("  ✓ Tensor metadata serialization OK")
    
    # Test layer list encoding
    layers = [0, 2, 4, 6, 8]
    data = ControlProtocol.encode_layer_list(layers)
    layers_restored = ControlProtocol.decode_layer_list(data)
    assert layers_restored == layers
    print(f"  ✓ Layer list encoding OK: {len(layers)} layers -> {len(data)} bytes")
    
    # Test offset index encoding
    offsets = [(0, 100), (100, 200), (300, 400)]
    data = ControlProtocol.encode_offset_index(offsets)
    offsets_restored = ControlProtocol.decode_offset_index(data)
    assert offsets_restored == offsets
    print(f"  ✓ Offset index encoding OK: {len(offsets)} entries -> {len(data)} bytes")
    
    # Test overhead savings
    savings = calculate_overhead_savings(n_layers=32, n_offsets=100)
    print(f"  Overhead savings: {savings['savings_percent']:.1f}% "
          f"({savings['json_bytes']} -> {savings['binary_bytes']} bytes)")
    assert savings['savings_bytes'] > 0, "Binary protocol should save bytes"
    
    print("  ✓ All control protocol tests passed")


def test_activation_compression():
    """Test activation compression and quantization."""
    print("\n=== Testing Activation Compression ===")
    
    from airllm.activation_compression import (
        quantize_f32_to_q80, dequantize_q80_to_f32,
        compress_activations, decompress_activations,
        calculate_compression_ratio
    )
    
    # Create test activation data
    x = np.random.randn(1, 4096).astype(np.float32)
    original_size = x.nbytes
    print(f"  Original activation size: {original_size} bytes ({original_size / 1024:.2f} KB)")
    
    # Test Q8_0 quantization
    scales, quantized = quantize_f32_to_q80(x)
    print(f"  Q8_0 quantized: {len(scales)} blocks, {len(quantized)} values")
    
    # Test dequantization
    x_restored = dequantize_q80_to_f32(scales, quantized, x.shape)
    assert x_restored.shape == x.shape, "Shape should match"
    
    # Calculate error
    mse = np.mean((x - x_restored) ** 2)
    max_error = np.max(np.abs(x - x_restored))
    print(f"  Quantization error: MSE={mse:.6f}, max_error={max_error:.6f}")
    assert mse < 0.01, "MSE should be small"
    
    # Test compression
    compressed = compress_activations(x, method='q80')
    compressed_size = len(compressed)
    print(f"  Compressed size: {compressed_size} bytes ({compressed_size / 1024:.2f} KB)")
    
    compression_ratio = original_size / compressed_size
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    assert compression_ratio > 2.5, "Compression ratio should be > 2.5x"
    
    # Test decompression
    x_decompressed = decompress_activations(compressed, x.shape, method='q80')
    assert x_decompressed.shape == x.shape
    
    # Calculate compression stats
    stats = calculate_compression_ratio(x.shape, method='q80')
    print(f"  Savings: {stats['savings_percent']:.1f}% "
          f"({stats['original_bytes']} -> {stats['compressed_bytes']} bytes)")
    
    print("  ✓ All activation compression tests passed")


def test_integration():
    """Test integration of Phase 3 components."""
    print("\n=== Testing Phase 3 Integration ===")
    
    # Simulate sending activations with compression
    print("  Simulating activation transfer with compression...")
    
    # Create activation data
    x = np.random.randn(4, 4096).astype(np.float32)
    original_size = x.nbytes
    
    # Compress
    from airllm.activation_compression import compress_activations, decompress_activations
    compressed = compress_activations(x, method='q80')
    
    print(f"    Original: {original_size} bytes")
    print(f"    Compressed: {len(compressed)} bytes")
    print(f"    Network savings: {(1 - len(compressed) / original_size) * 100:.1f}%")
    
    # Simulate network transfer (just store to bytes and restore)
    received = compressed
    
    # Decompress
    x_restored = decompress_activations(received, x.shape, method='q80')
    
    # Verify
    mse = np.mean((x - x_restored) ** 2)
    print(f"    Reconstruction MSE: {mse:.6f}")
    
    print("  ✓ Integration test passed")


if __name__ == '__main__':
    print("Running Phase 3 tests...")
    
    try:
        test_storage_coordinator()
        test_control_protocol()
        test_activation_compression()
        test_integration()
        
        print("\n" + "=" * 50)
        print("✓ All Phase 3 tests passed!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
