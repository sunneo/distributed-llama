"""
Activation compression and quantization for minimizing network transfer.

This module implements techniques to reduce the size of activation tensors
transferred between nodes:
- F32 to Q80 quantization (32-bit float to 8-bit quantized)
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
    Compress activation tensor for network transfer.
    
    Args:
        x: Input activation tensor (F32)
        method: Compression method ('q80', 'none')
        
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
    
    else:
        raise ValueError(f"Unsupported compression method: {method}")


def decompress_activations(data: bytes, shape: Tuple[int, ...], 
                          method: str = 'q80') -> np.ndarray:
    """
    Decompress activation tensor from network transfer.
    
    Args:
        data: Compressed binary data
        shape: Target output shape
        method: Compression method ('q80', 'none')
        
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
    
    else:
        raise ValueError(f"Unsupported compression method: {method}")


def calculate_compression_ratio(original_shape: Tuple[int, ...], 
                                method: str = 'q80',
                                block_size: int = 32) -> dict:
    """
    Calculate compression ratio for activation compression.
    
    Args:
        original_shape: Shape of F32 tensor
        method: Compression method
        block_size: Block size for quantization
        
    Returns:
        Dictionary with compression statistics
    """
    n_elements = np.prod(original_shape)
    original_bytes = n_elements * 4  # F32: 4 bytes per element
    
    if method == 'none':
        compressed_bytes = original_bytes
    elif method == 'q80':
        n_blocks = (n_elements + block_size - 1) // block_size
        header_bytes = 8  # 2 × uint32
        scales_bytes = n_blocks * 2  # float16 per block
        quantized_bytes = n_elements * 1  # int8 per element
        compressed_bytes = header_bytes + scales_bytes + quantized_bytes
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
