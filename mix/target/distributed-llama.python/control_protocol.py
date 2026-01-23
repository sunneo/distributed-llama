"""
Optimized control signal protocol for distributed inference.

This module implements a binary protocol for minimizing control signal overhead.
Instead of sending full metadata, we use compact binary formats for:
- Layer indices
- Weight offsets
- Tensor shapes
- Operation types
"""

import struct
from enum import IntEnum
from typing import Tuple, List, Optional
from dataclasses import dataclass


class OpType(IntEnum):
    """Operation types for worker tasks."""
    LOAD_LAYER = 1
    EXECUTE_LAYER = 2
    SYNC_ACTIVATIONS = 3
    FLUSH_CACHE = 4
    SHUTDOWN = 5


@dataclass
class ControlMessage:
    """Control message for worker coordination."""
    op_type: OpType
    layer_id: int = 0
    offset: int = 0
    size: int = 0
    seq_len: int = 0
    
    def to_bytes(self) -> bytes:
        """
        Serialize control message to binary format.
        
        Binary format (24 bytes):
        - op_type: uint32 (4 bytes)
        - layer_id: uint32 (4 bytes)
        - offset: uint64 (8 bytes)
        - size: uint32 (4 bytes)
        - seq_len: uint32 (4 bytes)
        
        Returns:
            Binary representation
        """
        return struct.pack('<IIQII',
                         self.op_type,
                         self.layer_id,
                         self.offset,
                         self.size,
                         self.seq_len)
    
    @staticmethod
    def from_bytes(data: bytes) -> 'ControlMessage':
        """
        Deserialize control message from binary format.
        
        Args:
            data: Binary data (must be at least 24 bytes)
            
        Returns:
            ControlMessage instance
        """
        if len(data) < 24:
            raise ValueError(f"Control message too short: {len(data)} bytes")
        
        op_type, layer_id, offset, size, seq_len = struct.unpack('<IIQII', data[:24])
        
        return ControlMessage(
            op_type=OpType(op_type),
            layer_id=layer_id,
            offset=offset,
            size=size,
            seq_len=seq_len
        )


@dataclass
class TensorMetadata:
    """Metadata for tensor transfer."""
    shape: Tuple[int, ...]
    dtype: str  # 'f32', 'q80', 'q40'
    
    def to_bytes(self) -> bytes:
        """
        Serialize tensor metadata to binary format.
        
        Binary format:
        - n_dims: uint8 (1 byte)
        - dims: n_dims × uint32 (4 bytes each)
        - dtype: 4 chars (4 bytes)
        
        Returns:
            Binary representation
        """
        n_dims = len(self.shape)
        
        # Pack number of dimensions
        data = struct.pack('<B', n_dims)
        
        # Pack each dimension
        for dim in self.shape:
            data += struct.pack('<I', dim)
        
        # Pack dtype as 4 chars (padded with null bytes)
        dtype_bytes = self.dtype.encode('ascii')[:4].ljust(4, b'\0')
        data += dtype_bytes
        
        return data
    
    @staticmethod
    def from_bytes(data: bytes) -> 'TensorMetadata':
        """
        Deserialize tensor metadata from binary format.
        
        Args:
            data: Binary data
            
        Returns:
            TensorMetadata instance
        """
        if len(data) < 5:
            raise ValueError(f"Tensor metadata too short: {len(data)} bytes")
        
        # Unpack number of dimensions
        n_dims = struct.unpack('<B', data[0:1])[0]
        
        if len(data) < 1 + n_dims * 4 + 4:
            raise ValueError(f"Tensor metadata incomplete")
        
        # Unpack dimensions
        shape = []
        offset = 1
        for _ in range(n_dims):
            dim = struct.unpack('<I', data[offset:offset+4])[0]
            shape.append(dim)
            offset += 4
        
        # Unpack dtype
        dtype_bytes = data[offset:offset+4]
        dtype = dtype_bytes.rstrip(b'\0').decode('ascii')
        
        return TensorMetadata(shape=tuple(shape), dtype=dtype)


class ControlProtocol:
    """
    Optimized control protocol for distributed inference.
    
    Minimizes overhead by using compact binary formats instead of JSON/text.
    """
    
    @staticmethod
    def encode_layer_list(layers: List[int]) -> bytes:
        """
        Encode list of layer IDs to binary format.
        
        Binary format:
        - n_layers: uint32 (4 bytes)
        - layer_ids: n_layers × uint32 (4 bytes each)
        
        Args:
            layers: List of layer indices
            
        Returns:
            Binary representation
        """
        data = struct.pack('<I', len(layers))
        for layer_id in layers:
            data += struct.pack('<I', layer_id)
        return data
    
    @staticmethod
    def decode_layer_list(data: bytes) -> List[int]:
        """
        Decode list of layer IDs from binary format.
        
        Args:
            data: Binary data
            
        Returns:
            List of layer indices
        """
        if len(data) < 4:
            raise ValueError("Layer list data too short")
        
        n_layers = struct.unpack('<I', data[0:4])[0]
        
        if len(data) < 4 + n_layers * 4:
            raise ValueError("Layer list data incomplete")
        
        layers = []
        offset = 4
        for _ in range(n_layers):
            layer_id = struct.unpack('<I', data[offset:offset+4])[0]
            layers.append(layer_id)
            offset += 4
        
        return layers
    
    @staticmethod
    def encode_offset_index(offsets: List[Tuple[int, int]]) -> bytes:
        """
        Encode list of (offset, size) pairs to binary format.
        
        Binary format:
        - n_items: uint32 (4 bytes)
        - items: n_items × (offset: uint64, size: uint32) (12 bytes each)
        
        Args:
            offsets: List of (offset, size) tuples
            
        Returns:
            Binary representation
        """
        data = struct.pack('<I', len(offsets))
        for offset, size in offsets:
            data += struct.pack('<QI', offset, size)
        return data
    
    @staticmethod
    def decode_offset_index(data: bytes) -> List[Tuple[int, int]]:
        """
        Decode list of (offset, size) pairs from binary format.
        
        Args:
            data: Binary data
            
        Returns:
            List of (offset, size) tuples
        """
        if len(data) < 4:
            raise ValueError("Offset index data too short")
        
        n_items = struct.unpack('<I', data[0:4])[0]
        
        if len(data) < 4 + n_items * 12:
            raise ValueError("Offset index data incomplete")
        
        offsets = []
        offset = 4
        for _ in range(n_items):
            off, size = struct.unpack('<QI', data[offset:offset+12])
            offsets.append((off, size))
            offset += 12
        
        return offsets


def calculate_overhead_savings(n_layers: int, n_offsets: int) -> dict:
    """
    Calculate overhead savings from using binary protocol.
    
    Estimated JSON overhead per entry:
    - Layer ID: ~20 bytes (e.g., '"layer_id": 123,\n')
    - Offset pair: ~50 bytes (e.g., '"offset": 123456, "size": 789,\n')
    - JSON structure: ~100 bytes (brackets, commas, quotes)
    
    Args:
        n_layers: Number of layers
        n_offsets: Number of offset entries
        
    Returns:
        Dictionary with overhead comparison
    """
    # Text/JSON protocol (estimated based on typical JSON serialization)
    json_layer_overhead = n_layers * 20  # Estimated: "layer_id": 123,
    json_offset_overhead = n_offsets * 50  # Estimated: "offset": 123456, "size": 789,
    json_total = json_layer_overhead + json_offset_overhead + 100  # JSON structure overhead
    
    # Binary protocol
    binary_layer_overhead = 4 + n_layers * 4
    binary_offset_overhead = 4 + n_offsets * 12
    binary_total = binary_layer_overhead + binary_offset_overhead
    
    return {
        'json_bytes': json_total,
        'binary_bytes': binary_total,
        'savings_bytes': json_total - binary_total,
        'savings_percent': ((json_total - binary_total) / json_total * 100) if json_total > 0 else 0,
        'compression_ratio': json_total / binary_total if binary_total > 0 else 0
    }
