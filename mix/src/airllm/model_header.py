"""
Model header parser for Distributed-Llama model files.

Parses the binary model header to extract architecture details
needed for layer-wise inference and weight loading.
"""

import struct
from dataclasses import dataclass
from typing import BinaryIO
from enum import IntEnum


# Header keys (matching C++ LlmHeaderKey enum)
class HeaderKey(IntEnum):
    VERSION = 0
    ARCH_TYPE = 1
    DIM = 2
    HIDDEN_DIM = 3
    N_LAYERS = 4
    N_HEADS = 5
    N_KV_HEADS = 6
    N_EXPERTS = 7
    N_ACTIVE_EXPERTS = 8
    VOCAB_SIZE = 9
    SEQ_LEN = 10
    HIDDEN_ACT = 11
    ROPE_THETA = 12
    WEIGHT_FLOAT_TYPE = 13
    ROPE_SCALING_FACTOR = 14
    ROPE_SCALING_LOW_FREQ_FACTOR = 15
    ROPE_SCALING_HIGH_FREQ_FACTORY = 16
    ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17
    ROPE_TYPE = 18
    HEAD_DIM = 19
    NORM_EPSILON = 20
    MOE_HIDDEN_DIM = 21


class ArchType(IntEnum):
    LLAMA = 0xABCD00
    QWEN3 = 0xABCD01
    QWEN3_MOE = 0xABCD02


class HiddenAct(IntEnum):
    GELU = 0
    SILU = 1


class RopeType(IntEnum):
    LLAMA = 0
    LLAMA3_1 = 1
    FALCON = 2


class FloatType(IntEnum):
    F32 = 0
    Q40 = 1
    Q80 = 2


@dataclass
class ModelHeader:
    """Model architecture configuration."""
    # Basic architecture
    version: int
    arch_type: ArchType
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    
    # Computed dimensions
    head_dim: int
    q_dim: int
    kv_dim: int
    
    # Activation and normalization
    hidden_act: HiddenAct
    norm_epsilon: float
    
    # RoPE configuration
    rope_type: RopeType
    rope_theta: float
    rope_scaling_factor: float
    rope_scaling_low_freq_factor: float
    rope_scaling_high_freq_factory: float
    rope_scaling_orig_max_seq_len: int
    
    # MoE (if applicable)
    n_experts: int
    n_active_experts: int
    moe_hidden_dim: int
    
    # Weight format
    weight_float_type: FloatType
    
    # File metadata
    header_size: int
    file_size: int


def convert_norm_epsilon(value: int) -> float:
    """Convert norm epsilon encoding to float."""
    if value == 5:
        return 1e-05
    if value == 6:
        return 1e-06
    raise ValueError(f"Unsupported norm epsilon: {value}")


def parse_model_header(model_path: str, max_seq_len: int = 0) -> ModelHeader:
    """
    Parse model header from distributed-llama model file.
    
    Args:
        model_path: Path to .m model file
        max_seq_len: Optional maximum sequence length override
        
    Returns:
        ModelHeader with architecture details
    """
    with open(model_path, 'rb') as f:
        # Read magic number (4 bytes)
        magic_bytes = f.read(4)
        magic = struct.unpack('<I', magic_bytes)[0]
        
        if magic == 0xABCD00 or magic == 0xABCD01:
            raise ValueError("Old model format is not supported")
        if magic != 0x0A00ABCD:
            raise ValueError(f"Unsupported magic number: 0x{magic:X}")
        
        # Read header size (4 bytes)
        header_size_bytes = f.read(4)
        header_size = struct.unpack('<I', header_size_bytes)[0]
        
        # Read header key-value pairs
        header_bytes = f.read(header_size)
        n_kv = (header_size - 8) // 4
        
        # Parse key-value pairs
        kv_data = struct.unpack(f'<{n_kv}i', header_bytes)
        
        # Initialize header with defaults
        header_dict = {
            'version': 0,
            'arch_type': ArchType.LLAMA,
            'dim': 0,
            'hidden_dim': 0,
            'n_layers': 0,
            'n_heads': 0,
            'n_kv_heads': 0,
            'vocab_size': 0,
            'seq_len': 0,
            'head_dim': 0,
            'hidden_act': HiddenAct.SILU,
            'norm_epsilon': 1e-5,
            'rope_type': RopeType.LLAMA,
            'rope_theta': 10000.0,
            'rope_scaling_factor': 1.0,
            'rope_scaling_low_freq_factor': 0.0,
            'rope_scaling_high_freq_factory': 0.0,
            'rope_scaling_orig_max_seq_len': 0,
            'n_experts': 0,
            'n_active_experts': 0,
            'moe_hidden_dim': 0,
            'weight_float_type': FloatType.F32,
            'header_size': header_size,
        }
        
        # Parse key-value pairs
        for i in range(0, n_kv, 2):
            key = kv_data[i]
            value = kv_data[i + 1]
            
            if key == HeaderKey.VERSION:
                header_dict['version'] = value
            elif key == HeaderKey.ARCH_TYPE:
                header_dict['arch_type'] = ArchType(value)
            elif key == HeaderKey.DIM:
                header_dict['dim'] = value
            elif key == HeaderKey.HIDDEN_DIM:
                header_dict['hidden_dim'] = value
            elif key == HeaderKey.N_LAYERS:
                header_dict['n_layers'] = value
            elif key == HeaderKey.N_HEADS:
                header_dict['n_heads'] = value
            elif key == HeaderKey.N_KV_HEADS:
                header_dict['n_kv_heads'] = value
            elif key == HeaderKey.N_EXPERTS:
                header_dict['n_experts'] = value
            elif key == HeaderKey.N_ACTIVE_EXPERTS:
                header_dict['n_active_experts'] = value
            elif key == HeaderKey.VOCAB_SIZE:
                header_dict['vocab_size'] = value
            elif key == HeaderKey.SEQ_LEN:
                header_dict['seq_len'] = value
            elif key == HeaderKey.HIDDEN_ACT:
                header_dict['hidden_act'] = HiddenAct(value)
            elif key == HeaderKey.ROPE_THETA:
                header_dict['rope_theta'] = float(value)
            elif key == HeaderKey.WEIGHT_FLOAT_TYPE:
                header_dict['weight_float_type'] = FloatType(value)
            elif key == HeaderKey.ROPE_SCALING_FACTOR:
                header_dict['rope_scaling_factor'] = float(value)
            elif key == HeaderKey.ROPE_SCALING_LOW_FREQ_FACTOR:
                header_dict['rope_scaling_low_freq_factor'] = float(value)
            elif key == HeaderKey.ROPE_SCALING_HIGH_FREQ_FACTORY:
                header_dict['rope_scaling_high_freq_factory'] = float(value)
            elif key == HeaderKey.ROPE_SCALING_ORIG_MAX_SEQ_LEN:
                header_dict['rope_scaling_orig_max_seq_len'] = value
            elif key == HeaderKey.ROPE_TYPE:
                header_dict['rope_type'] = RopeType(value)
            elif key == HeaderKey.HEAD_DIM:
                header_dict['head_dim'] = value
            elif key == HeaderKey.NORM_EPSILON:
                header_dict['norm_epsilon'] = convert_norm_epsilon(value)
            elif key == HeaderKey.MOE_HIDDEN_DIM:
                header_dict['moe_hidden_dim'] = value
        
        # Apply max_seq_len override if specified
        if max_seq_len > 0 and header_dict['seq_len'] > max_seq_len:
            header_dict['seq_len'] = max_seq_len
        
        # Calculate derived dimensions
        if header_dict['head_dim'] == 0:
            header_dict['head_dim'] = header_dict['dim'] // header_dict['n_heads']
        
        header_dict['q_dim'] = header_dict['head_dim'] * header_dict['n_heads']
        header_dict['kv_dim'] = header_dict['head_dim'] * header_dict['n_kv_heads']
        
        # Get file size
        f.seek(0, 2)  # Seek to end
        header_dict['file_size'] = f.tell()
        
        # Special handling for Qwen3
        if header_dict['arch_type'] in (ArchType.QWEN3, ArchType.QWEN3_MOE):
            header_dict['rope_type'] = RopeType.FALCON
        
        return ModelHeader(**header_dict)


def print_model_header(header: ModelHeader) -> None:
    """Print model header in human-readable format."""
    print(f"💡 Arch: {header.arch_type.name}")
    print(f"💡 HiddenAct: {header.hidden_act.name}")
    print(f"💡 Dim: {header.dim}")
    print(f"💡 HeadDim: {header.head_dim}")
    print(f"💡 QDim: {header.q_dim}")
    print(f"💡 KvDim: {header.kv_dim}")
    print(f"💡 HiddenDim: {header.hidden_dim}")
    print(f"💡 VocabSize: {header.vocab_size}")
    print(f"💡 nLayers: {header.n_layers}")
    print(f"💡 nHeads: {header.n_heads}")
    print(f"💡 nKvHeads: {header.n_kv_heads}")
    
    if header.n_experts > 0:
        print(f"💡 nExperts: {header.n_experts}")
        print(f"💡 nActiveExperts: {header.n_active_experts}")
        print(f"💡 MoeHiddenDim: {header.moe_hidden_dim}")
    
    print(f"💡 SeqLen: {header.seq_len}")
    print(f"💡 NormEpsilon: {header.norm_epsilon}")
    print(f"💡 RopeType: {header.rope_type.name}")
    print(f"💡 RopeTheta: {header.rope_theta:.0f}")
    print(f"💡 WeightType: {header.weight_float_type.name}")
    
    if header.rope_type == RopeType.LLAMA3_1:
        print(f"💡 RopeScaling: f={header.rope_scaling_factor:.1f}, "
              f"l={header.rope_scaling_low_freq_factor:.1f}, "
              f"h={header.rope_scaling_high_freq_factory:.1f}, "
              f"o={header.rope_scaling_orig_max_seq_len}")
