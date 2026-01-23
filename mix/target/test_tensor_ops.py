"""
Test tensor operations.

Basic unit tests for tensor operations to ensure correctness.
"""

import numpy as np
import sys
import os

# Add parent directory to path for development/testing
sys.path.insert(0, os.path.dirname(__file__))

from airllm import tensor_ops
from airllm.model_header import RopeType


def test_rms_norm():
    """Test RMS normalization."""
    print("Testing RMS normalization...")
    
    # Create test data
    x = np.random.randn(4, 8).astype(np.float32)
    weight = np.ones(8, dtype=np.float32)
    
    # Apply RMS norm
    output = tensor_ops.rms_norm(x, weight)
    
    # Check output shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    
    # Check that RMS is approximately 1
    rms = np.sqrt(np.mean(output * output, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-5), f"RMS not normalized: {rms}"
    
    print("✓ RMS normalization passed")


def test_matmul():
    """Test matrix multiplication."""
    print("Testing matrix multiplication...")
    
    # Create test data
    x = np.random.randn(4, 8).astype(np.float32)
    weight = np.random.randn(8, 16).astype(np.float32)
    
    # Apply matmul
    output = tensor_ops.matmul_f32(x, weight)
    
    # Check output shape
    assert output.shape == (4, 16), f"Shape mismatch: {output.shape} != (4, 16)"
    
    # Compare with numpy
    expected = np.matmul(x, weight)
    assert np.allclose(output, expected), "Matmul output incorrect"
    
    print("✓ Matrix multiplication passed")


def test_silu():
    """Test SiLU activation."""
    print("Testing SiLU activation...")
    
    # Create test data
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    
    # Apply SiLU
    output = tensor_ops.silu(x)
    
    # Check output shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    
    # Check some values
    # SiLU(0) should be 0
    assert np.abs(output[2]) < 1e-6, f"SiLU(0) = {output[2]}, expected 0"
    
    # SiLU(x) should be positive for positive x
    assert output[3] > 0, f"SiLU(1) = {output[3]}, expected > 0"
    assert output[4] > 0, f"SiLU(2) = {output[4]}, expected > 0"
    
    print("✓ SiLU activation passed")


def test_gelu():
    """Test GELU activation."""
    print("Testing GELU activation...")
    
    # Create test data
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    
    # Apply GELU
    output = tensor_ops.gelu(x)
    
    # Check output shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    
    # Check some values
    # GELU(0) should be approximately 0
    assert np.abs(output[2]) < 1e-2, f"GELU(0) = {output[2]}, expected ~0"
    
    # GELU(x) should be positive for positive x
    assert output[3] > 0, f"GELU(1) = {output[3]}, expected > 0"
    assert output[4] > 0, f"GELU(2) = {output[4]}, expected > 0"
    
    print("✓ GELU activation passed")


def test_rope():
    """Test RoPE (Rotary Position Embedding)."""
    print("Testing RoPE...")
    
    # Create test data
    seq_len = 4
    n_heads = 2
    head_dim = 8
    
    q = np.random.randn(seq_len, n_heads * head_dim).astype(np.float32)
    k = np.random.randn(seq_len, n_heads * head_dim).astype(np.float32)
    
    # Apply RoPE
    q_rot, k_rot = tensor_ops.apply_rope(q, k, pos=0, head_dim=head_dim)
    
    # Check output shapes
    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} != {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} != {k.shape}"
    
    print("✓ RoPE passed")


def test_attention():
    """Test multi-head attention."""
    print("Testing multi-head attention...")
    
    # Create test data
    seq_len = 4
    n_heads = 2
    n_kv_heads = 2
    head_dim = 8
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    
    q = np.random.randn(seq_len, q_dim).astype(np.float32)
    k = np.random.randn(seq_len, kv_dim).astype(np.float32)
    v = np.random.randn(seq_len, kv_dim).astype(np.float32)
    
    # Apply attention
    output = tensor_ops.multi_head_attention(q, k, v, n_heads, n_kv_heads)
    
    # Check output shape
    assert output.shape == (seq_len, q_dim), f"Shape mismatch: {output.shape} != {(seq_len, q_dim)}"
    
    print("✓ Multi-head attention passed")


def test_ffn():
    """Test feed-forward network."""
    print("Testing FFN...")
    
    # Create test data
    batch_size = 4
    dim = 8
    hidden_dim = 16
    
    x = np.random.randn(batch_size, dim).astype(np.float32)
    w1 = np.random.randn(dim, hidden_dim).astype(np.float32)
    w2 = np.random.randn(hidden_dim, dim).astype(np.float32)
    w3 = np.random.randn(dim, hidden_dim).astype(np.float32)
    
    # Apply FFN with SiLU
    output = tensor_ops.feed_forward(x, w1, w2, w3, activation='silu')
    
    # Check output shape
    assert output.shape == (batch_size, dim), f"Shape mismatch: {output.shape} != {(batch_size, dim)}"
    
    # Apply FFN with GELU
    output2 = tensor_ops.feed_forward(x, w1, w2, w3, activation='gelu')
    
    # Check output shape
    assert output2.shape == (batch_size, dim), f"Shape mismatch: {output2.shape} != {(batch_size, dim)}"
    
    print("✓ FFN passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running tensor operation tests...")
    print("=" * 60)
    
    test_rms_norm()
    test_matmul()
    test_silu()
    test_gelu()
    test_rope()
    test_attention()
    test_ffn()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
