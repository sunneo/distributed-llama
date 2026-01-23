# Distributed-Llama Python Worker

Python implementation of a worker node for the Distributed-Llama framework.

## Overview

This module allows Python to participate in distributed LLM inference alongside C++ nodes. It implements the same network protocol and tensor operations as the C++ workers, enabling heterogeneous clusters.

## Features

- ✅ Socket-based communication with C++ root node
- ✅ Binary protocol compatibility
- ✅ Configuration synchronization
- 🚧 Memory-mapped model weight loading (TODO)
- 🚧 Tensor operation execution (TODO)
- 🚧 Activation synchronization (TODO)

## Installation

```bash
cd mix/src/distributed-llama.python
pip install -r requirements.txt
```

## Usage

### As a Worker Node

```bash
python -m worker --host 192.168.1.100 --port 9999 --model /path/to/model.m
```

### In Python Code

```python
from distributed_llama_python import Worker

worker = Worker(host='192.168.1.100', port=9999, model_path='/path/to/model.m')
worker.connect()
worker.load_weights()
worker.run()
```

## Architecture

The Python worker implements these key components:

1. **NetworkClient** (`network.py`): TCP socket communication with root node
   - Sends/receives binary data
   - ACK protocol for synchronization
   - Chunked data transfer (4KB chunks)

2. **ConfigReader** (`config.py`): Reads network and node configuration
   - NetConfig: Global settings (nodes, batches, pipes)
   - NodeConfig: Worker-specific settings (segments, operations)

3. **Worker** (`worker.py`): Main worker implementation
   - Connects to root node
   - Allocates activation buffers
   - Executes tensor operations (TODO)
   - Synchronizes with other nodes (TODO)

## Protocol Compatibility

The Python implementation mirrors the C++ protocol:

```
Root (C++)  <-->  Worker (Python)
    |
    ├─> Send ACK
    ├─> Send NetConfig (batches, nodes, pipes)
    ├─> Receive ACK
    ├─> Send ACK
    ├─> Send NodeConfig (segments, ops, weights)
    ├─> Receive ACK
    └─> [Runtime: sync activations]
```

## TODO

See PLAN.md for detailed roadmap. Current TODOs:

- [ ] Implement memory-mapped weight loading with numpy.memmap
- [ ] Implement tensor operations (matmul, rope, attention, etc.)
- [ ] Implement activation synchronization protocol
- [ ] Add support for different float types (Q40, Q80, F32)
- [ ] Optimize critical paths with NumPy/native code
- [ ] Add comprehensive testing

## Development

```bash
# Run tests (TODO: add tests)
pytest

# Format code
black .

# Type checking
mypy .
```

## License

MIT (same as parent project)
