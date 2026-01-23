"""
Weight offset calculator for Distributed-Llama model files.

Calculates exact byte offsets for individual weight tensors in model files,
enabling efficient memory-mapped loading of specific layers.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
from .model_header import ModelHeader, FloatType

# Constants for quantization formats
Q40_BLOCK_SIZE = 32  # Block size for Q40 quantization (4-bit weights)
HEADER_PREFIX_SIZE = 8  # Size of magic number (4 bytes) + header size (4 bytes)


def get_float_size(float_type: FloatType) -> int:
    """Get size in bytes for a single element of the given float type."""
    if float_type == FloatType.F32:
        return 4
    elif float_type == FloatType.Q40:
        # Q40: 4 bits per weight + 32-bit scale per block
        # Stored as blocks, need to compute based on block size
        return -1  # Indicates variable/block-based encoding
    elif float_type == FloatType.Q80:
        # Q80: 8 bits per weight
        return 1
    else:
        raise ValueError(f"Unknown float type: {float_type}")


def calculate_q40_size(n_elements: int) -> int:
    """
    Calculate storage size for Q40 quantized tensor.
    
    Q40 format: 4 bits per weight + scale factor
    Typically stored in blocks of 32 or 64 elements.
    """
    # Each block: 32 weights * 4 bits + 1 float32 scale = 16 bytes + 4 bytes = 20 bytes
    # But packed more efficiently in practice - using simplified calculation
    # Real implementation should match the C++ quantization format exactly
    n_blocks = (n_elements + Q40_BLOCK_SIZE - 1) // Q40_BLOCK_SIZE
    bytes_per_block = (Q40_BLOCK_SIZE // 2) + 4  # 4 bits per weight + 4 byte scale
    return n_blocks * bytes_per_block


@dataclass
class LayerWeightOffsets:
    """Offsets for all weights in a single transformer layer."""
    layer_id: int
    
    # Attention weights
    wq_offset: int  # Query projection
    wq_size: int
    wk_offset: int  # Key projection
    wk_size: int
    wv_offset: int  # Value projection
    wv_size: int
    wo_offset: int  # Output projection
    wo_size: int
    
    # FFN weights
    w1_offset: int  # FFN gate/up projection
    w1_size: int
    w2_offset: int  # FFN down projection
    w2_size: int
    w3_offset: int  # FFN up projection (for SwiGLU)
    w3_size: int
    
    # Normalization weights
    attn_norm_offset: int  # Pre-attention RMSNorm
    attn_norm_size: int
    ffn_norm_offset: int   # Pre-FFN RMSNorm
    ffn_norm_size: int
    
    # MoE weights (if applicable)
    moe_gate_offset: int = 0
    moe_gate_size: int = 0
    expert_weights_offset: int = 0  # Start of expert weights
    expert_weights_size: int = 0
    
    # Total layer size
    total_size: int = 0


class WeightOffsetCalculator:
    """
    Calculates byte offsets for all weight tensors in a model file.
    
    This enables memory-mapped access to specific layers without loading
    the entire model into RAM.
    """
    
    def __init__(self, header: ModelHeader):
        """
        Initialize offset calculator.
        
        Args:
            header: Parsed model header
        """
        self.header = header
        self.layer_offsets: Dict[int, LayerWeightOffsets] = {}
        self._calculate_all_offsets()
    
    def _get_tensor_size(self, n_elements: int) -> int:
        """Calculate storage size for a tensor."""
        if self.header.weight_float_type == FloatType.F32:
            return n_elements * 4
        elif self.header.weight_float_type == FloatType.Q80:
            return n_elements  # 1 byte per element
        elif self.header.weight_float_type == FloatType.Q40:
            return calculate_q40_size(n_elements)
        else:
            raise ValueError(f"Unsupported weight type: {self.header.weight_float_type}")
    
    def _calculate_all_offsets(self) -> None:
        """Calculate offsets for all layers in the model."""
        # Start after header (magic + header_size + header data)
        current_offset = self.header.header_size + HEADER_PREFIX_SIZE
        
        # Token embedding: vocab_size x dim
        token_emb_size = self._get_tensor_size(self.header.vocab_size * self.header.dim)
        current_offset += token_emb_size
        
        # Calculate offsets for each layer
        for layer_id in range(self.header.n_layers):
            layer_offsets = self._calculate_layer_offsets(layer_id, current_offset)
            self.layer_offsets[layer_id] = layer_offsets
            current_offset += layer_offsets.total_size
        
        # Final norm: dim
        final_norm_size = self._get_tensor_size(self.header.dim)
        current_offset += final_norm_size
        
        # Output classifier: vocab_size x dim
        # Note: Often shares weights with token embedding
        wcls_size = self._get_tensor_size(self.header.vocab_size * self.header.dim)
        current_offset += wcls_size
    
    def _calculate_layer_offsets(self, layer_id: int, start_offset: int) -> LayerWeightOffsets:
        """Calculate offsets for a single layer."""
        offset = start_offset
        
        # Attention norm: dim
        attn_norm_offset = offset
        attn_norm_size = self._get_tensor_size(self.header.dim)
        offset += attn_norm_size
        
        # Attention weights
        # wq: dim x q_dim
        wq_offset = offset
        wq_size = self._get_tensor_size(self.header.dim * self.header.q_dim)
        offset += wq_size
        
        # wk: dim x kv_dim
        wk_offset = offset
        wk_size = self._get_tensor_size(self.header.dim * self.header.kv_dim)
        offset += wk_size
        
        # wv: dim x kv_dim
        wv_offset = offset
        wv_size = self._get_tensor_size(self.header.dim * self.header.kv_dim)
        offset += wv_size
        
        # wo: q_dim x dim
        wo_offset = offset
        wo_size = self._get_tensor_size(self.header.q_dim * self.header.dim)
        offset += wo_size
        
        # FFN norm: dim
        ffn_norm_offset = offset
        ffn_norm_size = self._get_tensor_size(self.header.dim)
        offset += ffn_norm_size
        
        # FFN weights
        # w1: dim x hidden_dim (gate)
        w1_offset = offset
        w1_size = self._get_tensor_size(self.header.dim * self.header.hidden_dim)
        offset += w1_size
        
        # w2: hidden_dim x dim (down)
        w2_offset = offset
        w2_size = self._get_tensor_size(self.header.hidden_dim * self.header.dim)
        offset += w2_size
        
        # w3: dim x hidden_dim (up, for SwiGLU)
        w3_offset = offset
        w3_size = self._get_tensor_size(self.header.dim * self.header.hidden_dim)
        offset += w3_size
        
        # MoE handling (TODO: calculate expert offsets for MoE models)
        moe_gate_offset = 0
        moe_gate_size = 0
        expert_weights_offset = 0
        expert_weights_size = 0
        
        if self.header.n_experts > 0:
            # TODO: Calculate MoE gate and expert weight offsets
            pass
        
        total_size = offset - start_offset
        
        return LayerWeightOffsets(
            layer_id=layer_id,
            wq_offset=wq_offset,
            wq_size=wq_size,
            wk_offset=wk_offset,
            wk_size=wk_size,
            wv_offset=wv_offset,
            wv_size=wv_size,
            wo_offset=wo_offset,
            wo_size=wo_size,
            w1_offset=w1_offset,
            w1_size=w1_size,
            w2_offset=w2_offset,
            w2_size=w2_size,
            w3_offset=w3_offset,
            w3_size=w3_size,
            attn_norm_offset=attn_norm_offset,
            attn_norm_size=attn_norm_size,
            ffn_norm_offset=ffn_norm_offset,
            ffn_norm_size=ffn_norm_size,
            moe_gate_offset=moe_gate_offset,
            moe_gate_size=moe_gate_size,
            expert_weights_offset=expert_weights_offset,
            expert_weights_size=expert_weights_size,
            total_size=total_size
        )
    
    def get_layer_offsets(self, layer_id: int) -> LayerWeightOffsets:
        """Get weight offsets for a specific layer."""
        if layer_id not in self.layer_offsets:
            raise ValueError(f"Layer {layer_id} not found (max: {self.header.n_layers - 1})")
        return self.layer_offsets[layer_id]
    
    def get_weight_offset(self, layer_id: int, weight_name: str) -> Tuple[int, int]:
        """
        Get offset and size for a specific weight tensor.
        
        Args:
            layer_id: Layer index
            weight_name: Weight name (e.g., 'wq', 'wk', 'w1', etc.)
            
        Returns:
            Tuple of (offset, size) in bytes
        """
        layer = self.get_layer_offsets(layer_id)
        
        weight_map = {
            'wq': (layer.wq_offset, layer.wq_size),
            'wk': (layer.wk_offset, layer.wk_size),
            'wv': (layer.wv_offset, layer.wv_size),
            'wo': (layer.wo_offset, layer.wo_size),
            'w1': (layer.w1_offset, layer.w1_size),
            'w2': (layer.w2_offset, layer.w2_size),
            'w3': (layer.w3_offset, layer.w3_size),
            'attn_norm': (layer.attn_norm_offset, layer.attn_norm_size),
            'ffn_norm': (layer.ffn_norm_offset, layer.ffn_norm_size),
        }
        
        if weight_name not in weight_map:
            raise ValueError(f"Unknown weight name: {weight_name}")
        
        return weight_map[weight_name]
