"""
AirLLM Layer-wise Inference Engine

This module implements layer-wise inference for running large models
on consumer hardware by swapping layers from disk.

Key concept: Only keep 1-2 layers in RAM at a time, load others from disk as needed.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from .model_header import ModelHeader, parse_model_header, print_model_header, HiddenAct
from .weight_offsets import WeightOffsetCalculator, LayerWeightOffsets
from . import tensor_ops
from .layer_cache import LayerCache


class MemoryMappedWeights:
    """
    Memory-mapped weight loader for zero-copy access.
    
    Uses numpy.memmap to access model weights without loading entire file.
    """
    
    def __init__(self, model_path: str, header: ModelHeader):
        """
        Initialize memory-mapped weights.
        
        Args:
            model_path: Path to model file
            header: Parsed model header
        """
        self.model_path = Path(model_path)
        self.header = header
        self.mmap: Optional[np.memmap] = None
        self.offset_calc = WeightOffsetCalculator(header)
        
    def open(self) -> None:
        """Open memory-mapped file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Open as read-only memory map
        self.mmap = np.memmap(self.model_path, dtype=np.uint8, mode='r')
        print(f"Opened memory-mapped file: {self.model_path} ({len(self.mmap)} bytes)")
    
    def close(self) -> None:
        """Close memory-mapped file."""
        if self.mmap is not None:
            del self.mmap
            self.mmap = None
    
    def load_weight_tensor(self, offset: int, size: int, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Load specific weight tensor from memory map.
        
        Args:
            offset: Byte offset in file
            size: Size in bytes
            shape: Desired tensor shape
            dtype: Data type of weights
            
        Returns:
            Numpy array with weights (view into mmap, zero-copy)
        """
        if self.mmap is None:
            raise RuntimeError("Memory map not opened")
        
        # Validate size is compatible with dtype
        itemsize = np.dtype(dtype).itemsize
        if size % itemsize != 0:
            raise ValueError(f"Size {size} is not evenly divisible by itemsize {itemsize}")
        
        # Create view into memory map
        n_elements = size // itemsize
        weights_flat = np.frombuffer(self.mmap[offset:offset + size], dtype=dtype, count=n_elements)
        
        # Reshape to desired shape
        try:
            weights = weights_flat.reshape(shape)
        except ValueError:
            # If reshape fails, return flat array
            print(f"Warning: Could not reshape to {shape}, returning flat array")
            weights = weights_flat
        
        return weights
    
    def load_layer_weights(self, layer_id: int) -> Dict[str, np.ndarray]:
        """
        Load all weight tensors for a specific layer.
        
        Args:
            layer_id: Layer index
            
        Returns:
            Dictionary mapping weight names to numpy arrays
        """
        layer_offsets = self.offset_calc.get_layer_offsets(layer_id)
        weights = {}
        
        # TODO: Handle quantized formats (Q40, Q80)
        # For now, assuming F32
        dtype = np.float32
        
        # Load attention weights
        wq_shape = (self.header.dim, self.header.q_dim)
        weights['wq'] = self.load_weight_tensor(
            layer_offsets.wq_offset, layer_offsets.wq_size, wq_shape, dtype
        )
        
        wk_shape = (self.header.dim, self.header.kv_dim)
        weights['wk'] = self.load_weight_tensor(
            layer_offsets.wk_offset, layer_offsets.wk_size, wk_shape, dtype
        )
        
        wv_shape = (self.header.dim, self.header.kv_dim)
        weights['wv'] = self.load_weight_tensor(
            layer_offsets.wv_offset, layer_offsets.wv_size, wv_shape, dtype
        )
        
        wo_shape = (self.header.q_dim, self.header.dim)
        weights['wo'] = self.load_weight_tensor(
            layer_offsets.wo_offset, layer_offsets.wo_size, wo_shape, dtype
        )
        
        # Load FFN weights
        w1_shape = (self.header.dim, self.header.hidden_dim)
        weights['w1'] = self.load_weight_tensor(
            layer_offsets.w1_offset, layer_offsets.w1_size, w1_shape, dtype
        )
        
        w2_shape = (self.header.hidden_dim, self.header.dim)
        weights['w2'] = self.load_weight_tensor(
            layer_offsets.w2_offset, layer_offsets.w2_size, w2_shape, dtype
        )
        
        w3_shape = (self.header.dim, self.header.hidden_dim)
        weights['w3'] = self.load_weight_tensor(
            layer_offsets.w3_offset, layer_offsets.w3_size, w3_shape, dtype
        )
        
        # Load normalization weights
        norm_shape = (self.header.dim,)
        weights['attn_norm'] = self.load_weight_tensor(
            layer_offsets.attn_norm_offset, layer_offsets.attn_norm_size, norm_shape, dtype
        )
        weights['ffn_norm'] = self.load_weight_tensor(
            layer_offsets.ffn_norm_offset, layer_offsets.ffn_norm_size, norm_shape, dtype
        )
        
        return weights


class LayerWiseInferenceEngine:
    """
    Layer-wise inference engine for AirLLM.
    
    Executes model layer-by-layer, swapping weights from disk.
    Only keeps current layer weights in RAM with LRU caching.
    """
    
    def __init__(self, model_path: str, cache_size: int = 2, max_memory_gb: float = 4.0):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file
            cache_size: Maximum number of layers to cache
            max_memory_gb: Maximum memory for cache in GB
        """
        self.model_path = model_path
        self.header: Optional[ModelHeader] = None
        self.weights_loader: Optional[MemoryMappedWeights] = None
        
        self.current_layer: Optional[int] = None
        self.current_weights: Optional[Dict[str, np.ndarray]] = None
        
        # Layer cache with LRU eviction
        self.cache = LayerCache(max_layers=cache_size, max_memory_gb=max_memory_gb)
    
    def initialize(self) -> None:
        """Initialize the engine by parsing model header."""
        # Parse model header
        self.header = parse_model_header(self.model_path)
        print_model_header(self.header)
        
        # Initialize memory-mapped weight loader
        self.weights_loader = MemoryMappedWeights(self.model_path, self.header)
        self.weights_loader.open()
        
        print(f"\nInitialized layer-wise engine for {self.header.n_layers} layers")
    
    def load_layer(self, layer_id: int, prefetch_next: bool = True) -> None:
        """
        Load weights for a specific layer with caching.
        
        Args:
            layer_id: Layer index to load
            prefetch_next: Whether to prefetch next layer
        """
        # Check cache first
        cached_weights = self.cache.get(layer_id)
        if cached_weights is not None:
            self.current_weights = cached_weights
            self.current_layer = layer_id
            print(f"Loaded layer {layer_id} from cache")
            
            # Prefetch next layer if requested
            if prefetch_next and layer_id + 1 < self.header.n_layers:
                self.cache.prefetch(layer_id + 1)
            
            return
        
        if self.weights_loader is None:
            raise RuntimeError("Engine not initialized")
        
        # Mark as loading
        self.cache.mark_loading(layer_id)
        
        # Load all weights for this layer from disk
        weights = self.weights_loader.load_layer_weights(layer_id)
        
        # Update cache
        self.cache.put(layer_id, weights)
        
        # Update current layer
        self.current_weights = weights
        self.current_layer = layer_id
        
        total_size = sum(w.nbytes for w in weights.values())
        print(f"Loaded layer {layer_id} from disk ({total_size / 1024 / 1024:.2f} MB)")
        
        # Prefetch next layer if requested
        if prefetch_next and layer_id + 1 < self.header.n_layers:
            self.cache.prefetch(layer_id + 1)
    
    def execute_layer(self, layer_id: int, x: np.ndarray, pos: int = 0,
                     kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Execute a single transformer layer.
        
        Args:
            layer_id: Layer index
            x: Input activations of shape (seq_len, dim)
            pos: Current position in sequence (for RoPE)
            kv_cache: Optional tuple of (key_cache, value_cache) for autoregressive generation
            
        Returns:
            Tuple of (output_activations, updated_kv_cache)
        """
        # Load layer weights if not already loaded
        self.load_layer(layer_id)
        
        if self.header is None or self.current_weights is None:
            raise RuntimeError("Layer not loaded")
        
        weights = self.current_weights
        
        # 1. Pre-attention RMS normalization
        x_norm = tensor_ops.rms_norm(x, weights['attn_norm'], eps=1e-6)
        
        # 2. Attention projections
        q = tensor_ops.matmul_f32(x_norm, weights['wq'])  # (seq_len, q_dim)
        k = tensor_ops.matmul_f32(x_norm, weights['wk'])  # (seq_len, kv_dim)
        v = tensor_ops.matmul_f32(x_norm, weights['wv'])  # (seq_len, kv_dim)
        
        # 3. Apply RoPE to Q and K
        q, k = tensor_ops.apply_rope(
            q, k, pos,
            head_dim=self.header.head_dim,
            rope_theta=self.header.rope_theta
        )
        
        # 4. Update KV cache if provided
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Append to cache
            k = np.concatenate([k_cache, k], axis=0)
            v = np.concatenate([v_cache, v], axis=0)
            kv_cache = (k, v)
        
        # 5. Multi-head attention
        attn_output = tensor_ops.multi_head_attention(
            q, k, v,
            n_heads=self.header.n_heads,
            n_kv_heads=self.header.n_kv_heads
        )
        
        # 6. Attention output projection
        attn_output = tensor_ops.matmul_f32(attn_output, weights['wo'])
        
        # 7. Residual connection
        x = x + attn_output
        
        # 8. Pre-FFN RMS normalization
        x_norm = tensor_ops.rms_norm(x, weights['ffn_norm'], eps=1e-6)
        
        # 9. Feed-forward network
        # Determine activation function
        activation = 'silu' if self.header.act == HiddenAct.SILU else 'gelu'
        ffn_output = tensor_ops.feed_forward(
            x_norm, weights['w1'], weights['w2'], weights['w3'],
            activation=activation
        )
        
        # 10. Residual connection
        x = x + ffn_output
        
        return x, kv_cache
    
    def forward(self, x: np.ndarray, start_pos: int = 0,
                use_cache: bool = False) -> np.ndarray:
        """
        Run forward pass through all layers.
        
        Args:
            x: Input tokens/embeddings of shape (seq_len, dim)
            start_pos: Starting position for RoPE
            use_cache: Whether to use KV caching for autoregressive generation
            
        Returns:
            Output activations of shape (seq_len, dim)
        """
        if self.header is None:
            raise RuntimeError("Engine not initialized")
        
        # Initialize KV cache if using cache
        kv_caches = [None] * self.header.n_layers
        
        for layer_id in range(self.header.n_layers):
            pos = start_pos  # Position for current token
            x, kv_caches[layer_id] = self.execute_layer(layer_id, x, pos, kv_caches[layer_id])
        
        return x
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.weights_loader:
            self.weights_loader.close()
        self.cache.clear()
