"""
Configuration data structures for Distributed-Llama Python worker.

These mirror the C++ structures (NnNetConfig, NnNodeConfig, etc.) to enable
protocol compatibility between Python and C++ nodes.
"""

import struct
from dataclasses import dataclass
from typing import List, Optional
from .network import NetworkClient


@dataclass
class PipeConfig:
    """Configuration for a data pipe (activation tensor buffer)."""
    size: int  # Size in bytes
    name: str


@dataclass
class PreSyncConfig:
    """Configuration for pre-synchronization."""
    pipe_index: int


@dataclass
class NetConfig:
    """Network-wide configuration."""
    n_batches: int
    n_nodes: int
    n_pipes: int
    pipes: List[PipeConfig]
    n_pre_syncs: int
    pre_syncs: List[PreSyncConfig]


@dataclass
class BufferConfig:
    """Configuration for a buffer."""
    size: int
    name: str


@dataclass
class SyncConfig:
    """Configuration for synchronization."""
    pipe_index: int
    sync_type: int


@dataclass
class OpConfig:
    """Configuration for an operation."""
    code: int  # OpCode enum value
    index: int
    weight_size: int
    config_size: int
    name: str
    input: int
    output: int
    config: Optional[bytes]


@dataclass
class SegmentConfig:
    """Configuration for a computation segment."""
    n_syncs: int
    n_ops: int
    syncs: List[SyncConfig]
    ops: List[OpConfig]


@dataclass
class NodeConfig:
    """Node-specific configuration."""
    node_index: int
    n_buffers: int
    n_segments: int
    buffers: List[BufferConfig]
    segments: List[SegmentConfig]


class ConfigReader:
    """
    Configuration reader for Python worker nodes.
    
    Reads network and node configuration from the root node,
    mirroring the C++ NnWorkerConfigReader class.
    """
    
    def __init__(self, network: NetworkClient):
        """
        Initialize config reader.
        
        Args:
            network: Network client connected to root node
        """
        self.network = network
    
    def _read_string(self) -> str:
        """Read a null-terminated string from network."""
        # Read string length (as uint32)
        len_bytes = self.network.read(4)
        str_len = struct.unpack('<I', len_bytes)[0]
        
        # Read string data
        if str_len > 0:
            str_bytes = self.network.read(str_len)
            # Remove null terminator if present
            return str_bytes.rstrip(b'\x00').decode('utf-8')
        return ""
    
    def read_net_config(self) -> NetConfig:
        """
        Read network configuration from root node.
        
        Returns:
            NetConfig object with network-wide settings
        """
        # Wait for ACK from root
        self.network.read_ack()
        
        # Read basic config
        n_batches = struct.unpack('<I', self.network.read(4))[0]
        n_nodes = struct.unpack('<I', self.network.read(4))[0]
        n_pipes = struct.unpack('<I', self.network.read(4))[0]
        
        # Read pipe configs
        pipes = []
        for _ in range(n_pipes):
            size = struct.unpack('<Q', self.network.read(8))[0]  # NnSize is uint64_t
            name = self._read_string()
            pipes.append(PipeConfig(size=size, name=name))
        
        # Read pre-sync configs
        n_pre_syncs = struct.unpack('<I', self.network.read(4))[0]
        pre_syncs = []
        for _ in range(n_pre_syncs):
            pipe_index = struct.unpack('<I', self.network.read(4))[0]
            pre_syncs.append(PreSyncConfig(pipe_index=pipe_index))
        
        # Send ACK to root
        self.network.write_ack()
        
        return NetConfig(
            n_batches=n_batches,
            n_nodes=n_nodes,
            n_pipes=n_pipes,
            pipes=pipes,
            n_pre_syncs=n_pre_syncs,
            pre_syncs=pre_syncs
        )
    
    def read_node_config(self) -> NodeConfig:
        """
        Read node-specific configuration from root node.
        
        Returns:
            NodeConfig object with this worker's settings
        """
        # Wait for ACK from root
        self.network.read_ack()
        
        # Read basic config
        node_index = struct.unpack('<I', self.network.read(4))[0]
        n_buffers = struct.unpack('<I', self.network.read(4))[0]
        n_segments = struct.unpack('<I', self.network.read(4))[0]
        
        # Read buffer configs
        buffers = []
        for _ in range(n_buffers):
            size = struct.unpack('<Q', self.network.read(8))[0]
            name = self._read_string()
            buffers.append(BufferConfig(size=size, name=name))
        
        # Read segment configs
        segments = []
        for _ in range(n_segments):
            n_syncs = struct.unpack('<I', self.network.read(4))[0]
            n_ops = struct.unpack('<I', self.network.read(4))[0]
            
            # Read sync configs
            syncs = []
            for _ in range(n_syncs):
                pipe_index = struct.unpack('<I', self.network.read(4))[0]
                sync_type = struct.unpack('<I', self.network.read(4))[0]
                syncs.append(SyncConfig(pipe_index=pipe_index, sync_type=sync_type))
            
            # Read op configs
            ops = []
            for _ in range(n_ops):
                code = struct.unpack('<I', self.network.read(4))[0]
                index = struct.unpack('<I', self.network.read(4))[0]
                weight_size = struct.unpack('<Q', self.network.read(8))[0]
                config_size = struct.unpack('<Q', self.network.read(8))[0]
                name = self._read_string()
                input_idx = struct.unpack('<I', self.network.read(4))[0]
                output_idx = struct.unpack('<I', self.network.read(4))[0]
                
                # Read op config data if present
                config_data = None
                if config_size > 0:
                    config_data = self.network.read(config_size)
                
                ops.append(OpConfig(
                    code=code,
                    index=index,
                    weight_size=weight_size,
                    config_size=config_size,
                    name=name,
                    input=input_idx,
                    output=output_idx,
                    config=config_data
                ))
            
            segments.append(SegmentConfig(
                n_syncs=n_syncs,
                n_ops=n_ops,
                syncs=syncs,
                ops=ops
            ))
        
        # Send ACK to root
        self.network.write_ack()
        
        return NodeConfig(
            node_index=node_index,
            n_buffers=n_buffers,
            n_segments=n_segments,
            buffers=buffers,
            segments=segments
        )
