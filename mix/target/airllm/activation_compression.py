"""
Activation compression and quantization for minimizing network transfer
and KV cache disk I/O.

This module implements techniques to reduce the size of activation tensors
transferred between nodes, and to compress KV cache entries before writing
to disk/SSD:
- F32 to Q80 quantization (32-bit float to 8-bit quantized, ~4x compression)
- F32 to Q40 quantization (32-bit float to 4-bit quantized, ~7x compression,
  recommended for KV cache disk offloading as it reduces I/O by ~75-87%)
- Simple compression for sparse activations
"""

import numpy as np
from typing import Tuple, Optional


def quantize_f32_to_q80(x: np.ndarray, block_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize F32 activations to Q8_0 format.
    
    Q8_0 format: 8-bit quantization with blocks of values sharing a scale factor.
    Each block has: 1 float16 scale + block_size int8 values
    
    This reduces network traffic by ~4x (32-bit -> 8-bit + overhead).
    
    Args:
        x: Input tensor in F32 format
        block_size: Number of values per quantization block (default 32)
        
    Returns:
        Tuple of (scales, quantized_values)
        - scales: Float16 array of scale factors, shape (n_blocks,)
        - quantized_values: Int8 array of quantized values, shape (n_elements,)
    """
    # Flatten input
    original_shape = x.shape
    x_flat = x.flatten()
    n_elements = len(x_flat)
    
    # Calculate number of blocks
    n_blocks = (n_elements + block_size - 1) // block_size
    
    # Allocate output arrays
    scales = np.zeros(n_blocks, dtype=np.float16)
    quantized = np.zeros(n_elements, dtype=np.int8)
    
    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, n_elements)
        block = x_flat[start_idx:end_idx]
        
        # Calculate scale: max absolute value in block
        abs_max = np.max(np.abs(block))
        
        # Avoid division by zero
        # Note: We use 127.0 (not 128) to avoid saturation at the edge of int8 range.
        # This provides symmetric quantization: [-127, 127] maps to [-abs_max, abs_max]
        if abs_max == 0:
            scale = 0.0
        else:
            scale = abs_max / 127.0  # Scale to fit in int8 range [-127, 127]
        
        scales[block_idx] = scale
        
        # Quantize block
        if scale > 0:
            quantized_block = np.round(block / scale).astype(np.int8)
            # Clamp to int8 range
            quantized_block = np.clip(quantized_block, -127, 127)
            quantized[start_idx:end_idx] = quantized_block
    
    return scales, quantized


def dequantize_q80_to_f32(scales: np.ndarray, quantized: np.ndarray, 
                          shape: Tuple[int, ...], block_size: int = 32) -> np.ndarray:
    """
    Dequantize Q8_0 format back to F32.
    
    Args:
        scales: Float16 array of scale factors
        quantized: Int8 array of quantized values
        shape: Target output shape
        block_size: Number of values per quantization block
        
    Returns:
        Dequantized F32 tensor
    """
    n_elements = len(quantized)
    n_blocks = len(scales)
    
    # Allocate output
    output = np.zeros(n_elements, dtype=np.float32)
    
    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, n_elements)
        
        scale = float(scales[block_idx])
        quantized_block = quantized[start_idx:end_idx]
        
        # Dequantize: multiply by scale
        output[start_idx:end_idx] = quantized_block.astype(np.float32) * scale
    
    return output.reshape(shape)


def quantize_f32_to_q40(x: np.ndarray, block_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize F32 tensor to Q4_0 format (4-bit symmetric quantization).

    Q4_0 format: 4-bit quantization with blocks of values sharing a scale.
    Each block has: 1 float16 scale + block_size/2 bytes of packed nibbles.

    Reduces size by ~87% vs F32 (>75% I/O savings when writing to SSD).
    Recommended for KV cache disk offloading to minimise SSD write pressure.

    Args:
        x: Input tensor in F32 format
        block_size: Number of values per quantization block (must be even,
                    default 32)

    Returns:
        Tuple of (scales, quantized_packed)
        - scales: Float16 array of scale factors, shape (n_blocks,)
        - quantized_packed: Uint8 array of packed 4-bit values (two nibbles per
          byte), shape (n_blocks * block_size // 2,)
    """
    if block_size % 2 != 0:
        raise ValueError(f"block_size must be even, got {block_size}")

    x_flat = x.flatten()
    n_elements = len(x_flat)
    n_blocks = (n_elements + block_size - 1) // block_size

    # Pad to an exact multiple of block_size
    padded_len = n_blocks * block_size
    x_padded = np.zeros(padded_len, dtype=np.float32)
    x_padded[:n_elements] = x_flat
    x_blocks = x_padded.reshape(n_blocks, block_size)

    # Scale: abs-max per block, mapped to symmetric [-7, 7] range
    abs_max = np.max(np.abs(x_blocks), axis=1)  # (n_blocks,)
    scales = np.where(abs_max > 0, abs_max / 7.0, 0.0).astype(np.float16)

    # Quantize each block to [-8, 7] (clamped to fit in a signed nibble)
    safe_scales = np.where(scales.astype(np.float32) > 0,
                           scales.astype(np.float32), 1.0)
    q = np.round(x_blocks / safe_scales[:, np.newaxis]).astype(np.int32)
    q = np.clip(q, -8, 7)
    q[scales == 0] = 0  # zero out blocks whose scale is zero

    # Shift to unsigned [0, 15] and pack two nibbles per byte
    q_unsigned = (q + 8).astype(np.uint8)           # (n_blocks, block_size)
    q_even = q_unsigned[:, 0::2]                     # even-indexed values
    q_odd  = q_unsigned[:, 1::2]                     # odd-indexed values
    quantized_packed = (q_even | (q_odd << 4)).reshape(-1)  # (n_blocks * block_size // 2,)

    return scales, quantized_packed


def dequantize_q40_to_f32(scales: np.ndarray, quantized_packed: np.ndarray,
                           shape: Tuple[int, ...], block_size: int = 32) -> np.ndarray:
    """
    Dequantize Q4_0 format back to F32.

    Args:
        scales: Float16 array of scale factors, shape (n_blocks,)
        quantized_packed: Uint8 array of packed nibbles,
                          shape (n_blocks * block_size // 2,)
        shape: Target output shape
        block_size: Number of values per quantization block

    Returns:
        Dequantized F32 tensor with the given shape
    """
    n_blocks = len(scales)
    n_pairs = len(quantized_packed)  # = n_blocks * block_size // 2

    # Unpack nibbles: low nibble → even indices, high nibble → odd indices
    low_nibbles  = (quantized_packed & 0x0F).astype(np.uint8)
    high_nibbles = ((quantized_packed >> 4) & 0x0F).astype(np.uint8)

    q_unsigned = np.empty(n_pairs * 2, dtype=np.uint8)
    q_unsigned[0::2] = low_nibbles
    q_unsigned[1::2] = high_nibbles

    # Shift back to signed [-8, 7]
    q_signed = q_unsigned.astype(np.int32) - 8  # (n_blocks * block_size,)
    q_blocks  = q_signed.reshape(n_blocks, block_size)

    # Dequantize
    output = (q_blocks * scales[:, np.newaxis].astype(np.float32)).reshape(-1)

    # Trim padding and reshape to original dimensions
    n_elements = int(np.prod(shape))
    return output[:n_elements].reshape(shape)


def pack_q40_for_storage(scales: np.ndarray, quantized_packed: np.ndarray) -> bytes:
    """
    Pack Q4_0 quantized data into compact binary format for SSD storage.

    Binary format:
    - n_blocks: uint32 (4 bytes)
    - n_packed: uint32 (4 bytes)   — number of packed-nibble bytes
    - scales: n_blocks × float16 (2 bytes each)
    - quantized_packed: n_packed × uint8 (1 byte each)

    Args:
        scales: Scale factors array (float16)
        quantized_packed: Packed nibble values (uint8)

    Returns:
        Binary packed data
    """
    n_blocks = len(scales)
    n_packed = len(quantized_packed)

    header = np.array([n_blocks, n_packed], dtype=np.uint32)
    return b''.join([header.tobytes(), scales.tobytes(), quantized_packed.tobytes()])


def unpack_q40_from_storage(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unpack Q4_0 quantized data from binary storage format.

    Args:
        data: Binary packed data (produced by pack_q40_for_storage)

    Returns:
        Tuple of (scales, quantized_packed)
    """
    header = np.frombuffer(data[0:8], dtype=np.uint32)
    n_blocks = int(header[0])
    n_packed = int(header[1])

    scales_start = 8
    scales_end   = scales_start + n_blocks * 2
    scales = np.frombuffer(data[scales_start:scales_end], dtype=np.float16).copy()

    packed_start = scales_end
    packed_end   = packed_start + n_packed
    quantized_packed = np.frombuffer(data[packed_start:packed_end], dtype=np.uint8).copy()

    return scales, quantized_packed


def pack_q80_for_transfer(scales: np.ndarray, quantized: np.ndarray) -> bytes:
    """
    Pack Q8_0 quantized data into compact binary format for network transfer.
    
    Binary format:
    - n_blocks: uint32 (4 bytes)
    - n_elements: uint32 (4 bytes)
    - scales: n_blocks × float16 (2 bytes each)
    - quantized: n_elements × int8 (1 byte each)
    
    Args:
        scales: Scale factors array
        quantized: Quantized values array
        
    Returns:
        Binary packed data
    """
    n_blocks = len(scales)
    n_elements = len(quantized)
    
    # Pack header
    header = np.array([n_blocks, n_elements], dtype=np.uint32)
    
    # Concatenate all data
    data = b''.join([
        header.tobytes(),
        scales.tobytes(),
        quantized.tobytes()
    ])
    
    return data


def unpack_q80_from_transfer(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unpack Q8_0 quantized data from network transfer format.
    
    Args:
        data: Binary packed data
        
    Returns:
        Tuple of (scales, quantized)
    """
    # Read header
    header = np.frombuffer(data[0:8], dtype=np.uint32)
    n_blocks = int(header[0])
    n_elements = int(header[1])
    
    # Read scales
    scales_start = 8
    scales_end = scales_start + n_blocks * 2
    scales = np.frombuffer(data[scales_start:scales_end], dtype=np.float16)
    
    # Read quantized values
    quantized_start = scales_end
    quantized_end = quantized_start + n_elements
    quantized = np.frombuffer(data[quantized_start:quantized_end], dtype=np.int8)
    
    return scales, quantized


def compress_activations(x: np.ndarray, method: str = 'q80') -> bytes:
    """
    Compress activation tensor for network transfer or disk storage.

    Args:
        x: Input activation tensor (F32)
        method: Compression method ('q80', 'q40', 'none')

    Returns:
        Compressed binary data
    """
    if method == 'none':
        # No compression, just convert to bytes
        return x.tobytes()

    elif method == 'q80':
        # Quantize to Q8_0 format
        scales, quantized = quantize_f32_to_q80(x)
        return pack_q80_for_transfer(scales, quantized)

    elif method == 'q40':
        # Quantize to Q4_0 format (recommended for KV cache disk offloading)
        scales, quantized_packed = quantize_f32_to_q40(x)
        return pack_q40_for_storage(scales, quantized_packed)

    else:
        raise ValueError(f"Unsupported compression method: {method}")


def decompress_activations(data: bytes, shape: Tuple[int, ...],
                           method: str = 'q80') -> np.ndarray:
    """
    Decompress activation tensor from network transfer or disk storage.

    Args:
        data: Compressed binary data
        shape: Target output shape
        method: Compression method ('q80', 'q40', 'none')

    Returns:
        Decompressed F32 tensor
    """
    if method == 'none':
        # No compression, just convert from bytes
        return np.frombuffer(data, dtype=np.float32).reshape(shape)

    elif method == 'q80':
        # Dequantize from Q8_0 format
        scales, quantized = unpack_q80_from_transfer(data)
        return dequantize_q80_to_f32(scales, quantized, shape)

    elif method == 'q40':
        # Dequantize from Q4_0 format
        scales, quantized_packed = unpack_q40_from_storage(data)
        return dequantize_q40_to_f32(scales, quantized_packed, shape)

    else:
        raise ValueError(f"Unsupported compression method: {method}")


def calculate_compression_ratio(original_shape: Tuple[int, ...],
                                method: str = 'q80',
                                block_size: int = 32) -> dict:
    """
    Calculate compression ratio for activation or KV cache compression.

    Args:
        original_shape: Shape of F32 tensor
        method: Compression method ('none', 'q80', 'q40')
        block_size: Block size for quantization

    Returns:
        Dictionary with compression statistics
    """
    n_elements = int(np.prod(original_shape))
    original_bytes = n_elements * 4  # F32: 4 bytes per element

    if method == 'none':
        compressed_bytes = original_bytes
    elif method == 'q80':
        n_blocks = (n_elements + block_size - 1) // block_size
        header_bytes = 8  # 2 × uint32
        scales_bytes = n_blocks * 2  # float16 per block
        quantized_bytes = n_elements * 1  # int8 per element
        compressed_bytes = header_bytes + scales_bytes + quantized_bytes
    elif method == 'q40':
        n_blocks = (n_elements + block_size - 1) // block_size
        padded_elements = n_blocks * block_size
        header_bytes = 8  # 2 × uint32
        scales_bytes = n_blocks * 2  # float16 per block
        packed_bytes = padded_elements // 2  # 2 nibbles per byte
        compressed_bytes = header_bytes + scales_bytes + packed_bytes
    else:
        compressed_bytes = original_bytes

    return {
        'original_bytes': original_bytes,
        'compressed_bytes': compressed_bytes,
        'savings_bytes': original_bytes - compressed_bytes,
        'compression_ratio': original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0,
        'savings_percent': ((original_bytes - compressed_bytes) / original_bytes * 100) if original_bytes > 0 else 0
    }


def benchmark_compression(shape: Tuple[int, ...] = (1, 4096), 
                         n_iterations: int = 100) -> dict:
    """
    Benchmark compression performance.
    
    Args:
        shape: Tensor shape to test
        n_iterations: Number of iterations for timing
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Generate random activation data
    x = np.random.randn(*shape).astype(np.float32)
    
    # Benchmark no compression
    start = time.time()
    for _ in range(n_iterations):
        data = x.tobytes()
        x_restored = np.frombuffer(data, dtype=np.float32).reshape(shape)
    time_none = time.time() - start
    
    # Benchmark Q8_0 compression
    start = time.time()
    for _ in range(n_iterations):
        data = compress_activations(x, method='q80')
        x_restored = decompress_activations(data, shape, method='q80')
    time_q80 = time.time() - start
    
    # Calculate MSE for Q8_0
    data = compress_activations(x, method='q80')
    x_restored = decompress_activations(data, shape, method='q80')
    mse = np.mean((x - x_restored) ** 2)
    
    # Get compression stats
    stats_q80 = calculate_compression_ratio(shape, method='q80')
    
    return {
        'shape': shape,
        'iterations': n_iterations,
        'time_none_sec': time_none,
        'time_q80_sec': time_q80,
        'throughput_none_mb_per_sec': (np.prod(shape) * 4 * n_iterations / time_none) / (1024 ** 2),
        'throughput_q80_mb_per_sec': (np.prod(shape) * 4 * n_iterations / time_q80) / (1024 ** 2),
        'q80_mse': float(mse),
        'compression_ratio': stats_q80['compression_ratio'],
        'savings_percent': stats_q80['savings_percent']
    }
