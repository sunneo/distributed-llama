# Tensor Operations Module

This module provides the core tensor operations needed for transformer layer execution:

## Features

### 1. Normalization
- **RMS Normalization**: Layer normalization used in modern transformers (LLAMA, etc.)

### 2. Matrix Operations
- **Matrix Multiplication**: Supports F32, Q40, and Q80 quantized weights
- **Dequantization**: Convert Q40 and Q80 quantized weights to F32

### 3. Position Encoding
- **RoPE (Rotary Position Embedding)**: Positional encoding without explicit position embeddings
- Supports LLAMA, FALCON, and LLAMA3.1 variants

### 4. Attention
- **Multi-Head Attention**: Standard transformer attention mechanism
- **Grouped Query Attention (GQA)**: Memory-efficient attention with fewer KV heads
- Supports causal masking

### 5. Activation Functions
- **SiLU (Sigmoid Linear Unit)**: Used in LLAMA models
- **GELU (Gaussian Error Linear Unit)**: Used in GPT models

### 6. Feed-Forward Networks
- **SwiGLU FFN**: Feed-forward network with gated activation
- Supports both SiLU and GELU activations

## Usage Example

```python
import numpy as np
from airllm import tensor_ops

# RMS Normalization
x = np.random.randn(4, 128).astype(np.float32)
weight = np.ones(128, dtype=np.float32)
x_norm = tensor_ops.rms_norm(x, weight, eps=1e-6)

# Matrix Multiplication
x = np.random.randn(4, 128).astype(np.float32)
weight = np.random.randn(128, 256).astype(np.float32)
output = tensor_ops.matmul_f32(x, weight)

# RoPE
q = np.random.randn(4, 512).astype(np.float32)  # (seq_len, q_dim)
k = np.random.randn(4, 512).astype(np.float32)  # (seq_len, kv_dim)
q_rot, k_rot = tensor_ops.apply_rope(q, k, pos=0, head_dim=64)

# Multi-Head Attention
q = np.random.randn(4, 512).astype(np.float32)  # 8 heads x 64 dim
k = np.random.randn(4, 512).astype(np.float32)
v = np.random.randn(4, 512).astype(np.float32)
output = tensor_ops.multi_head_attention(q, k, v, n_heads=8, n_kv_heads=8)

# FFN
x = np.random.randn(4, 128).astype(np.float32)
w1 = np.random.randn(128, 512).astype(np.float32)
w2 = np.random.randn(512, 128).astype(np.float32)
w3 = np.random.randn(128, 512).astype(np.float32)
output = tensor_ops.feed_forward(x, w1, w2, w3, activation='silu')
```

## Testing

Run the unit tests:

```bash
cd mix/target
python3 test_tensor_ops.py
```

All tests should pass:
- ✓ RMS normalization
- ✓ Matrix multiplication
- ✓ SiLU activation
- ✓ GELU activation
- ✓ RoPE
- ✓ Multi-head attention
- ✓ FFN

## Quantization Support

The module supports quantized weight formats:
- **Q40**: 4-bit quantization (16 values per block, 2-byte scale)
- **Q80**: 8-bit quantization (32 values per block, 2-byte scale)

Quantized weights are automatically dequantized before computation.

## Performance Notes

- All operations use NumPy for CPU computation
- For production use, consider:
  - Vectorization with AVX/AVX512
  - GPU acceleration with CuPy/CUDA
  - C++ implementation with pybind11 bindings (Phase 4)

## References

- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [LLAMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [GQA: Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
