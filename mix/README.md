# Distributed-AirLLM: Merged Implementation

This directory contains the merged implementation of **Distributed-Llama** and **AirLLM** concepts, creating a system for running large language models (30B+) on distributed consumer hardware.

## Core Concept: "Shared-Storage Zero-Data Movement"

Instead of distributing model weights across nodes:
- ✅ Every node has the **full model on local SSD**
- ✅ Each node loads **only its assigned layers** into RAM
- ✅ Nodes transmit **only activations** over the network (not weights)
- ✅ Network traffic reduced from GBs to KBs per token

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Root Node (C++)                   │
│  - Orchestrates inference                           │
│  - Loads layers 0-7                                 │
│  - Sends activations to workers                     │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌───────▼────────┐
│ Worker 1 (Py)  │  │ Worker 2 (Py)  │
│ Layers 8-15    │  │ Layers 16-23   │
│ Model on SSD   │  │ Model on SSD   │
└────────────────┘  └────────────────┘

All nodes: Same 30B model file on local storage
Network: Only activation tensors (~few KB per token)
```

## Components

### 1. Distributed-Llama Python Worker (`distributed-llama.python/`)

Python implementation of a worker node compatible with the C++ root node.

**Features:**
- ✅ Binary protocol compatibility (socket, ACK, chunked I/O)
- ✅ Config reader (NetConfig, NodeConfig)
- ✅ Activation buffer management (pipes)
- 🚧 Tensor operation execution (TODO)
- 🚧 Synchronization protocol (TODO)

**Files:**
- `network.py`: TCP socket communication
- `config.py`: Configuration data structures
- `worker.py`: Main worker loop
- `README.md`: Documentation

### 2. AirLLM Layer-wise Engine (`airllm/`)

Layer-wise inference engine for memory-efficient model execution.

**Features:**
- ✅ Model header parser (LLAMA, QWEN3, QWEN3_MOE)
- ✅ Weight offset calculator (F32, Q40, Q80)
- ✅ Memory-mapped weight loading (zero-copy)
- ✅ Per-layer weight access
- 🚧 Tensor operations (TODO)
- 🚧 Layer caching (TODO)

**Files:**
- `model_header.py`: Binary header parser
- `weight_offsets.py`: Byte offset calculator
- `layer_engine.py`: Layer-wise execution engine
- `README.md`: Documentation
- `examples/parse_header.py`: Example usage

## Current Status

### ✅ Completed (Phase 1 & 2 - 55%)

1. **Python Worker Framework**
   - Socket communication with C++ root
   - Configuration synchronization
   - Worker lifecycle management

2. **Model Header Parsing**
   - Binary format parser
   - Support for multiple architectures
   - Quantization format handling

3. **Weight Offset Calculation**
   - Exact byte offsets for all tensors
   - Per-layer and per-weight access
   - Memory-mapped loading support

### 🚧 In Progress (Next Steps)

1. **Tensor Operations** (Phase 1.3)
   - RMS normalization
   - Matrix multiplication
   - RoPE (Rotary Position Embedding)
   - Multi-head attention
   - FFN (Feed-forward network)

2. **Activation Synchronization** (Phase 1.4)
   - Receive activations from root
   - Send activations back
   - Handle sync protocol

3. **Layer Caching** (Phase 2.4)
   - LRU cache for hot layers
   - Prefetching strategy
   - Memory pressure management

### 📋 Planned (Phases 3-4)

1. **Zero-Data Movement Optimizations** (Phase 3)
   - Shared storage verification
   - Control signal optimization
   - Activation compression

2. **C++ Bottleneck Rewrite** (Phase 4)
   - Profile Python implementation
   - Rewrite hot paths in C++
   - Create pybind11 bindings

## Usage Examples

### Parse Model Header

```bash
cd mix/src/airllm
python examples/parse_header.py /path/to/model.m
```

### Run Python Worker (when complete)

```bash
cd mix/src/distributed-llama.python
python -m worker --host 192.168.1.100 --port 9999 --model /path/to/model.m
```

### Expected Workflow (when complete)

```bash
# Terminal 1: Start root node (C++)
./dllama inference --model model.m --workers 192.168.1.2:9999 192.168.1.3:9999

# Terminal 2: Start Python worker 1
python -m worker --host 192.168.1.1 --port 9999 --model /mnt/ssd/model.m

# Terminal 3: Start Python worker 2
python -m worker --host 192.168.1.1 --port 9999 --model /mnt/ssd/model.m
```

## Benefits Over Standard Distributed Inference

| Aspect | Traditional | Distributed-AirLLM |
|--------|------------|---------------------|
| Model Storage | Sharded across nodes | Full model on each node |
| RAM Usage | Full shard in RAM | Only assigned layers |
| Network Traffic | Weights + activations | Activations only |
| Node Addition | Requires rebalancing | Just add worker |
| Fault Tolerance | Lose shard = failure | Any node can load any layer |
| Storage Cost | N × (Model/N) | N × Model |

## Testing

```bash
# Install dependencies
pip install -r distributed-llama.python/requirements.txt

# Run header parser test (requires actual model file)
python airllm/examples/parse_header.py /path/to/model.m
```

## Development Roadmap

See [`PLAN.md`](PLAN.md) for detailed task tracking.

**Current Phase:** 2 - AirLLM Integration (60% complete)

**Next Milestone:** Implement tensor operations to enable actual inference

## Technical Details

### Binary Protocol

The Python worker implements the same protocol as C++ workers:

```
1. Connect to root
2. Receive ACK
3. Receive NetConfig (batches, nodes, pipes)
4. Send ACK
5. Receive ACK
6. Receive NodeConfig (segments, ops)
7. Send ACK
8. Main loop:
   - Receive sync signal
   - Execute ops
   - Send results
```

### Model File Format

Distributed-llama uses a custom binary format:

```
[Magic: 0x0A00ABCD]
[Header Size: uint32]
[Key-Value Pairs: (key, value) tuples]
[Token Embedding Weights]
[Layer 0 Weights]
[Layer 1 Weights]
...
[Layer N Weights]
[Final Norm]
[Output Classifier]
```

### Weight Offset Calculation

For each layer:
```
offset = header_end + token_emb_size + sum(previous_layer_sizes)

Layer contents:
- Attention norm (dim floats)
- wq, wk, wv, wo (attention weights)
- FFN norm (dim floats)
- w1, w2, w3 (FFN weights)
```

## Contributing

When implementing new features:

1. Mark TODOs in code and `PLAN.md`
2. Update progress in `PLAN.md` 
3. Add examples in `examples/`
4. Document in relevant README

## License

MIT (same as parent project)
