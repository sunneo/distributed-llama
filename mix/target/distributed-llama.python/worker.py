"""
Worker node implementation for Distributed-Llama in Python.

This module implements the main worker loop that:
1. Connects to the root node
2. Receives configuration
3. Loads model weights (memory-mapped)
4. Executes assigned tensor operations
5. Synchronizes activations with other nodes
"""

import numpy as np
from typing import Optional, Dict, List
from .network import NetworkClient
from .config import ConfigReader, NetConfig, NodeConfig
import sys
import os

# Add parent directory to path to import airllm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from airllm.layer_engine import LayerWiseInferenceEngine


class Worker:
    """
    Distributed-Llama Python Worker Node.
    
    This worker connects to a C++ root node and executes its assigned
    portion of the neural network inference.
    """
    
    def __init__(self, host: str, port: int, model_path: Optional[str] = None):
        """
        Initialize worker.
        
        Args:
            host: Root node hostname/IP
            port: Root node port
            model_path: Path to model file (for weight loading)
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        
        self.network: Optional[NetworkClient] = None
        self.net_config: Optional[NetConfig] = None
        self.node_config: Optional[NodeConfig] = None
        
        # Activation buffers (pipes)
        self.pipes: Dict[int, np.ndarray] = {}
        
        # Model weights (layer-wise inference engine)
        self.inference_engine: Optional[LayerWiseInferenceEngine] = None
        
        # Assigned layers for this worker
        self.assigned_layers: List[int] = []
        
    def connect(self) -> None:
        """Connect to root node and receive configuration."""
        print(f"Connecting to root node at {self.host}:{self.port}...")
        
        self.network = NetworkClient(self.host, self.port)
        self.network.connect()
        
        # Read configuration from root
        config_reader = ConfigReader(self.network)
        
        print("Reading network configuration...")
        self.net_config = config_reader.read_net_config()
        print(f"  Nodes: {self.net_config.n_nodes}")
        print(f"  Batches: {self.net_config.n_batches}")
        print(f"  Pipes: {self.net_config.n_pipes}")
        
        print("Reading node configuration...")
        self.node_config = config_reader.read_node_config()
        print(f"  Node index: {self.node_config.node_index}")
        print(f"  Segments: {self.node_config.n_segments}")
        print(f"  Buffers: {self.node_config.n_buffers}")
        
        # Allocate activation buffers (pipes)
        self._allocate_pipes()
        
        print("Worker initialization complete")
    
    def _allocate_pipes(self) -> None:
        """Allocate activation buffers based on pipe configuration."""
        print("Allocating activation pipes...")
        for i, pipe_config in enumerate(self.net_config.pipes):
            # Allocate as float32 buffer for now
            # TODO: Support different float types (q80, f32, etc.)
            n_floats = pipe_config.size // 4  # Assuming 4 bytes per float32
            self.pipes[i] = np.zeros(n_floats, dtype=np.float32)
            print(f"  Pipe {i} ({pipe_config.name}): {pipe_config.size} bytes")
    
    def load_weights(self) -> None:
        """
        Load model weights using layer-wise inference engine.
        
        Initializes the engine with memory-mapped weights and determines
        which layers this worker is responsible for.
        """
        if not self.model_path:
            print("WARNING: No model path specified, weights not loaded")
            return
        
        print(f"Initializing layer-wise inference engine: {self.model_path}")
        
        # Initialize layer-wise inference engine
        self.inference_engine = LayerWiseInferenceEngine(self.model_path)
        self.inference_engine.initialize()
        
        # Determine assigned layers based on node configuration
        # TODO: Implement proper layer distribution across nodes
        # For now, assign all layers to all workers (shared-storage model)
        if self.inference_engine.header:
            total_layers = self.inference_engine.header.n_layers
            n_nodes = self.net_config.n_nodes if self.net_config else 1
            node_idx = self.node_config.node_index if self.node_config else 0
            
            # Simple round-robin distribution
            self.assigned_layers = list(range(node_idx, total_layers, n_nodes))
            print(f"Assigned layers: {self.assigned_layers}")
    
    def receive_activations(self, pipe_index: int) -> np.ndarray:
        """
        Receive activations from root or previous node.
        
        Args:
            pipe_index: Index of pipe to receive from
            
        Returns:
            Received activation tensor
        """
        if self.network is None:
            raise RuntimeError("Not connected to network")
        
        # TODO: Implement proper activation receive protocol
        # This should match the C++ synchronization protocol
        pipe_buffer = self.pipes[pipe_index]
        
        # Read activations from network
        data = self.network.receive_bytes(pipe_buffer.nbytes)
        
        # Convert to numpy array
        activations = np.frombuffer(data, dtype=pipe_buffer.dtype).reshape(pipe_buffer.shape)
        
        return activations
    
    def send_activations(self, pipe_index: int, activations: np.ndarray) -> None:
        """
        Send activations to root or next node.
        
        Args:
            pipe_index: Index of pipe to send to
            activations: Activation tensor to send
        """
        if self.network is None:
            raise RuntimeError("Not connected to network")
        
        # TODO: Implement proper activation send protocol
        # This should match the C++ synchronization protocol
        
        # Convert to bytes and send
        data = activations.tobytes()
        self.network.send_bytes(data)
    
    def run(self) -> None:
        """
        Main worker loop.
        
        Executes assigned layers and synchronizes activations with other nodes.
        """
        if self.inference_engine is None:
            raise RuntimeError("Inference engine not initialized")
        
        print("Starting worker main loop...")
        print(f"Assigned layers: {self.assigned_layers}")
        
        # Placeholder for main loop
        import time
        try:
            iteration = 0
            while True:
                # TODO: Wait for work signal from root node
                # TODO: Receive input activations
                # TODO: Execute assigned layers
                # TODO: Send output activations
                
                # Placeholder: Just wait
                time.sleep(1.0)
                
                iteration += 1
                if iteration % 10 == 0:
                    print(f"Worker iteration {iteration}, waiting for work...")
                    
        except KeyboardInterrupt:
            print("\nShutdown requested")
        except Exception as e:
            print(f"Error in worker loop: {e}")
            raise
    
    def shutdown(self) -> None:
        """Shutdown worker and disconnect from root."""
        if self.inference_engine:
            self.inference_engine.cleanup()
        if self.network:
            self.network.disconnect()
        print("Worker shutdown complete")


def main():
    """Command-line entry point for worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed-Llama Python Worker')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Root node hostname/IP')
    parser.add_argument('--port', type=int, default=9999,
                        help='Root node port')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file')
    
    args = parser.parse_args()
    
    worker = Worker(args.host, args.port, args.model)
    
    try:
        worker.connect()
        worker.load_weights()
        worker.run()
    finally:
        worker.shutdown()


if __name__ == '__main__':
    main()
