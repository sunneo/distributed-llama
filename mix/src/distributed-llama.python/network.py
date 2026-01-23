"""
Network communication layer for Distributed-Llama Python worker.

Implements the socket protocol compatible with the C++ root node.
"""

import socket
import struct
from typing import Optional, Tuple

# Protocol constants matching C++ implementation (src/nn/nn-network.cpp)
ACK = 23571114  # Acknowledgment packet value
MAX_CHUNK_SIZE = 4096  # Maximum bytes per socket I/O operation


class NetworkException(Exception):
    """Base exception for network errors."""
    pass


class ConnectionException(NetworkException):
    """Exception raised when connection fails."""
    pass


class TransferException(NetworkException):
    """Exception raised when data transfer fails."""
    pass


class NetworkClient:
    """
    Network client for connecting to the root node.
    
    This implements the same socket protocol as the C++ NnNetwork class,
    allowing Python workers to communicate with C++ root nodes.
    """
    
    def __init__(self, host: str, port: int):
        """
        Initialize network client.
        
        Args:
            host: Root node hostname/IP
            port: Root node port
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.sent_bytes = 0
        self.recv_bytes = 0
        
    def connect(self) -> None:
        """Connect to the root node."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Enable TCP_NODELAY for low-latency communication
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.connect((self.host, self.port))
            print(f"Connected to root node at {self.host}:{self.port}")
        except Exception as e:
            raise ConnectionException(f"Failed to connect to {self.host}:{self.port}: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from the root node."""
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Disconnected from root node")
    
    def write(self, data: bytes) -> None:
        """
        Write data to the socket.
        
        Args:
            data: Bytes to send
        """
        if not self.socket:
            raise NetworkException("Not connected")
        
        size = len(data)
        offset = 0
        
        # Send in chunks to match C++ implementation
        while offset < size:
            chunk_size = min(MAX_CHUNK_SIZE, size - offset)
            chunk = data[offset:offset + chunk_size]
            
            try:
                sent = self.socket.send(chunk)
                if sent == 0:
                    raise TransferException("Socket closed during write")
                offset += sent
                self.sent_bytes += sent
            except socket.error as e:
                raise TransferException(f"Error writing to socket: {e}")
    
    def read(self, size: int) -> bytes:
        """
        Read data from the socket.
        
        Args:
            size: Number of bytes to read
            
        Returns:
            Bytes read from socket
        """
        if not self.socket:
            raise NetworkException("Not connected")
        
        data = bytearray()
        
        # Read in chunks to match C++ implementation
        while len(data) < size:
            chunk_size = min(MAX_CHUNK_SIZE, size - len(data))
            
            try:
                chunk = self.socket.recv(chunk_size)
                if not chunk:
                    raise TransferException("Socket closed during read")
                data.extend(chunk)
                self.recv_bytes += len(chunk)
            except socket.error as e:
                raise TransferException(f"Error reading from socket: {e}")
        
        return bytes(data)
    
    def write_ack(self) -> None:
        """Write acknowledgment packet."""
        ack_bytes = struct.pack('<I', ACK)  # Little-endian unsigned int
        self.write(ack_bytes)
    
    def read_ack(self) -> None:
        """Read acknowledgment packet."""
        ack_bytes = self.read(4)
        ack = struct.unpack('<I', ack_bytes)[0]
        if ack != ACK:
            raise TransferException(f"Invalid ACK packet: expected {ACK}, got {ack}")
    
    def get_stats(self) -> Tuple[int, int]:
        """
        Get network statistics.
        
        Returns:
            Tuple of (sent_bytes, recv_bytes)
        """
        sent = self.sent_bytes
        recv = self.recv_bytes
        self.sent_bytes = 0
        self.recv_bytes = 0
        return (sent, recv)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
