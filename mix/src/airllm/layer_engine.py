"""
AirLLM Layer-wise Inference Engine

This module implements layer-wise inference for running large models
on consumer hardware by swapping layers from disk.

Key concept: Only keep 1-2 layers in RAM at a time, load others from disk as needed.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from .model_header import ModelHeader, parse_model_header, print_model_header
from .weight_offsets import WeightOffsetCalculator, LayerWeightOffsets


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
        
        # Create view into memory map
        n_elements = size // np.dtype(dtype).itemsize
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
    Only keeps current layer weights in RAM.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file
        """
        self.model_path = model_path
        self.header: Optional[ModelHeader] = None
        self.weights_loader: Optional[MemoryMappedWeights] = None
        
        self.current_layer: Optional[int] = None
        self.current_weights: Optional[Dict[str, np.ndarray]] = None
    
    def initialize(self) -> None:
        """Initialize the engine by parsing model header."""
        # Parse model header
        self.header = parse_model_header(self.model_path)
        print_model_header(self.header)
        
        # Initialize memory-mapped weight loader
        self.weights_loader = MemoryMappedWeights(self.model_path, self.header)
        self.weights_loader.open()
        
        print(f"\nInitialized layer-wise engine for {self.header.n_layers} layers")
    
    def load_layer(self, layer_id: int) -> None:
        """
        Load weights for a specific layer.
        
        Args:
            layer_id: Layer index to load
        """
        if self.current_layer == layer_id and self.current_weights is not None:
            return  # Already loaded
        
        if self.weights_loader is None:
            raise RuntimeError("Engine not initialized")
        
        # Load all weights for this layer
        self.current_weights = self.weights_loader.load_layer_weights(layer_id)
        self.current_layer = layer_id
        
        total_size = sum(w.nbytes for w in self.current_weights.values())
        print(f"Loaded layer {layer_id} ({total_size / 1024 / 1024:.2f} MB)")
    
    def execute_layer(self, layer_id: int, x: np.ndarray) -> np.ndarray:
        """
        Execute a single transformer layer.
        
        Args:
            layer_id: Layer index
            x: Input activations
            
        Returns:
            Output activations
        """
        # Load layer weights if not already loaded
        self.load_layer(layer_id)
        
        # TODO: Implement actual layer computation
        # - RMS normalization
        # - Attention (Q, K, V, O projections + multi-head attention)
        # - FFN (W1, W2, W3 with SiLU/GELU activation)
        
        print(f"TODO: Execute layer {layer_id}")
        return x  # Placeholder
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run forward pass through all layers.
        
        Args:
            x: Input tokens/embeddings
            
        Returns:
            Output activations
        """
        if self.header is None:
            raise RuntimeError("Engine not initialized")
        
        for layer_id in range(self.header.n_layers):
            x = self.execute_layer(layer_id, x)
        
        return x
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.weights_loader:
            self.weights_loader.close()
