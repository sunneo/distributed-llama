"""
Tensor operations for AirLLM layer-wise inference.

This module implements the core tensor operations needed for transformer layers:
- RMS normalization
- Matrix multiplication (with quantization support)
- RoPE (Rotary Position Embedding)
- Multi-head attention
- Activation functions (SiLU, GELU)
- Feed-forward network (FFN)
"""

import numpy as np
from typing import Tuple, Optional
from .model_header import ModelHeader, RopeType


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply RMS (Root Mean Square) normalization.
    
    RMS norm is defined as:
        output = x * weight / sqrt(mean(x^2) + eps)
    
    Args:
        x: Input tensor of shape (..., dim)
        weight: Normalization weights of shape (dim,)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor of same shape as input
    """
    # Compute RMS: sqrt(mean(x^2))
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    
    # Normalize: x / rms
    x_normalized = x / rms
    
    # Scale by learned weights
    return x_normalized * weight


def matmul_f32(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication for F32 weights.
    
    Args:
        x: Input tensor of shape (..., in_dim)
        weight: Weight matrix of shape (in_dim, out_dim)
        
    Returns:
        Output tensor of shape (..., out_dim)
    """
    return np.matmul(x, weight)


def dequantize_q40(data: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Dequantize Q4_0 format to F32.
    
    Q4_0 format: 4-bit quantization with 16 values per block
    Each block has: 1 float32 scale + 16 int4 values (8 bytes)
    Block size: 20 bytes (4 + 16)
    
    Args:
        data: Raw quantized data as uint8 array
        shape: Target shape for output tensor
        
    Returns:
        Dequantized F32 tensor
    """
    # Q4_0 block structure
    block_size = 32  # 32 values per block
    bytes_per_block = 18  # 2 (scale as fp16) + 16 (packed 4-bit values)
    
    n_elements = np.prod(shape)
    n_blocks = (n_elements + block_size - 1) // block_size
    
    output = np.zeros(n_elements, dtype=np.float32)
    
    for block_idx in range(n_blocks):
        block_offset = block_idx * bytes_per_block
        
        # Read scale (fp16)
        scale_bytes = data[block_offset:block_offset + 2]
        scale = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)
        
        # Read quantized values (16 bytes = 32 4-bit values)
        quant_offset = block_offset + 2
        quant_bytes = data[quant_offset:quant_offset + 16]
        
        # Unpack 4-bit values
        for i in range(16):
            byte_val = quant_bytes[i]
            # Lower 4 bits
            val0 = (byte_val & 0x0F) - 8  # Signed 4-bit: 0-15 -> -8 to 7
            # Upper 4 bits
            val1 = ((byte_val >> 4) & 0x0F) - 8
            
            out_idx0 = block_idx * block_size + i * 2
            out_idx1 = out_idx0 + 1
            
            if out_idx0 < n_elements:
                output[out_idx0] = val0 * scale
            if out_idx1 < n_elements:
                output[out_idx1] = val1 * scale
    
    return output[:n_elements].reshape(shape)


def dequantize_q80(data: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Dequantize Q8_0 format to F32.
    
    Q8_0 format: 8-bit quantization with 32 values per block
    Each block has: 1 float32 scale + 32 int8 values
    Block size: 36 bytes (4 + 32)
    
    Args:
        data: Raw quantized data as uint8 array
        shape: Target shape for output tensor
        
    Returns:
        Dequantized F32 tensor
    """
    # Q8_0 block structure
    block_size = 32  # 32 values per block
    bytes_per_block = 34  # 2 (scale as fp16) + 32 (int8 values)
    
    n_elements = np.prod(shape)
    n_blocks = (n_elements + block_size - 1) // block_size
    
    output = np.zeros(n_elements, dtype=np.float32)
    
    for block_idx in range(n_blocks):
        block_offset = block_idx * bytes_per_block
        
        # Read scale (fp16)
        scale_bytes = data[block_offset:block_offset + 2]
        scale = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)
        
        # Read quantized values (int8)
        quant_offset = block_offset + 2
        quant_values = np.frombuffer(
            data[quant_offset:quant_offset + 32],
            dtype=np.int8
        )
        
        # Dequantize
        out_start = block_idx * block_size
        out_end = min(out_start + block_size, n_elements)
        out_slice = slice(out_start, out_end)
        
        output[out_slice] = quant_values[:out_end - out_start].astype(np.float32) * scale
    
    return output[:n_elements].reshape(shape)


def matmul_quantized(x: np.ndarray, weight_data: np.ndarray, 
                     weight_shape: Tuple[int, ...], 
                     quant_type: str) -> np.ndarray:
    """
    Matrix multiplication with quantized weights.
    
    Args:
        x: Input tensor of shape (..., in_dim)
        weight_data: Quantized weight data as uint8 array
        weight_shape: Shape of weight matrix (in_dim, out_dim)
        quant_type: Quantization type ('Q40' or 'Q80')
        
    Returns:
        Output tensor of shape (..., out_dim)
    """
    # Dequantize weights
    if quant_type == 'Q40':
        weight = dequantize_q40(weight_data, weight_shape)
    elif quant_type == 'Q80':
        weight = dequantize_q80(weight_data, weight_shape)
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")
    
    # Perform matrix multiplication
    return np.matmul(x, weight)


def apply_rope(q: np.ndarray, k: np.ndarray, pos: int, 
               head_dim: int, rope_theta: float = 10000.0,
               rope_type: RopeType = RopeType.LLAMA) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Rotary Position Embedding (RoPE) to query and key tensors.
    
    RoPE applies rotation to pairs of features in the embedding dimension.
    This encodes positional information without explicit position embeddings.
    
    Args:
        q: Query tensor of shape (seq_len, q_dim)
        k: Key tensor of shape (seq_len, kv_dim)
        pos: Current position in sequence
        head_dim: Dimension of each attention head
        rope_theta: Theta parameter for rotation frequency
        rope_type: Type of RoPE (LLAMA, FALCON, etc.)
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Compute rotation frequencies for each dimension pair
    # freq[i] = 1 / (theta^(2i/d)) for i = 0, 1, ..., d/2-1
    dim_pairs = head_dim // 2
    freqs = 1.0 / (rope_theta ** (np.arange(0, dim_pairs, dtype=np.float32) * 2 / head_dim))
    
    # Compute rotation angles at current position
    angles = pos * freqs  # shape: (dim_pairs,)
    
    # Compute cos and sin
    cos = np.cos(angles)
    sin = np.sin(angles)
    
    def rotate(x: np.ndarray) -> np.ndarray:
        """Apply rotation to tensor."""
        # x shape: (seq_len, total_dim)
        seq_len, total_dim = x.shape
        n_heads_local = total_dim // head_dim
        
        # Reshape to separate heads and dimensions
        x_reshape = x.reshape(seq_len, n_heads_local, head_dim)
        
        # Further reshape to separate even/odd indices
        x_reshape = x_reshape.reshape(seq_len, n_heads_local, -1, 2)  # (seq_len, n_heads, dim_pairs, 2)
        
        # Extract even and odd
        x_even = x_reshape[..., 0]  # (seq_len, n_heads, dim_pairs)
        x_odd = x_reshape[..., 1]   # (seq_len, n_heads, dim_pairs)
        
        # Apply rotation: [cos, -sin; sin, cos]
        # Broadcast cos and sin to match shape
        x_even_rot = x_even * cos[None, None, :] - x_odd * sin[None, None, :]
        x_odd_rot = x_even * sin[None, None, :] + x_odd * cos[None, None, :]
        
        # Interleave back
        x_rot = np.stack([x_even_rot, x_odd_rot], axis=-1)
        x_rot = x_rot.reshape(seq_len, n_heads_local, head_dim)
        return x_rot.reshape(seq_len, total_dim)
    
    q_rot = rotate(q)
    k_rot = rotate(k)
    
    return q_rot, k_rot


def multi_head_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                         n_heads: int, n_kv_heads: int,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Supports Grouped Query Attention (GQA) where n_kv_heads < n_heads.
    
    Args:
        q: Query tensor of shape (seq_len, q_dim) where q_dim = n_heads * head_dim
        k: Key tensor of shape (kv_seq_len, kv_dim) where kv_dim = n_kv_heads * head_dim
        v: Value tensor of shape (kv_seq_len, kv_dim)
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads
        mask: Optional attention mask
        
    Returns:
        Output tensor of shape (seq_len, q_dim)
    """
    seq_len = q.shape[0]
    kv_seq_len = k.shape[0]
    head_dim = q.shape[1] // n_heads
    
    # Reshape to separate heads
    q = q.reshape(seq_len, n_heads, head_dim)  # (seq_len, n_heads, head_dim)
    k = k.reshape(kv_seq_len, n_kv_heads, head_dim)  # (kv_seq_len, n_kv_heads, head_dim)
    v = v.reshape(kv_seq_len, n_kv_heads, head_dim)  # (kv_seq_len, n_kv_heads, head_dim)
    
    # Handle GQA: repeat k, v to match number of query heads
    if n_kv_heads < n_heads:
        n_rep = n_heads // n_kv_heads
        k = np.repeat(k, n_rep, axis=1)  # (kv_seq_len, n_heads, head_dim)
        v = np.repeat(v, n_rep, axis=1)  # (kv_seq_len, n_heads, head_dim)
    
    # Transpose for batch matrix multiplication
    q = q.transpose(1, 0, 2)  # (n_heads, seq_len, head_dim)
    k = k.transpose(1, 0, 2)  # (n_heads, kv_seq_len, head_dim)
    v = v.transpose(1, 0, 2)  # (n_heads, kv_seq_len, head_dim)
    
    # Compute attention scores: Q @ K^T / sqrt(head_dim)
    scores = np.matmul(q, k.transpose(0, 2, 1))  # (n_heads, seq_len, kv_seq_len)
    scores = scores / np.sqrt(head_dim)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + mask
    
    # Softmax over key dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
    attn_weights = scores_exp / scores_sum  # (n_heads, seq_len, kv_seq_len)
    
    # Apply attention to values: attn @ V
    output = np.matmul(attn_weights, v)  # (n_heads, seq_len, head_dim)
    
    # Transpose back and reshape
    output = output.transpose(1, 0, 2)  # (seq_len, n_heads, head_dim)
    output = output.reshape(seq_len, n_heads * head_dim)  # (seq_len, q_dim)
    
    return output


def silu(x: np.ndarray) -> np.ndarray:
    """
    Apply SiLU (Sigmoid Linear Unit) activation function.
    
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor of same shape
    """
    return x / (1.0 + np.exp(-x))


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Apply GELU (Gaussian Error Linear Unit) activation function.
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor of same shape
    """
    # Use tanh approximation for efficiency
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(inner))


def feed_forward(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, 
                 w3: np.ndarray, activation: str = 'silu') -> np.ndarray:
    """
    Apply feed-forward network (FFN) layer.
    
    FFN architecture (SwiGLU variant):
        FFN(x) = W2 @ (activation(W1 @ x) ⊙ (W3 @ x))
    
    Where ⊙ is element-wise multiplication.
    
    Args:
        x: Input tensor of shape (..., dim)
        w1: First projection weight of shape (dim, hidden_dim)
        w2: Second projection weight of shape (hidden_dim, dim)
        w3: Gate projection weight of shape (dim, hidden_dim)
        activation: Activation function ('silu' or 'gelu')
        
    Returns:
        Output tensor of shape (..., dim)
    """
    # First projection and activation
    h1 = matmul_f32(x, w1)
    if activation == 'silu':
        h1 = silu(h1)
    elif activation == 'gelu':
        h1 = gelu(h1)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    # Gate projection
    h3 = matmul_f32(x, w3)
    
    # Element-wise multiplication
    h = h1 * h3
    
    # Second projection
    output = matmul_f32(h, w2)
    
    return output
