# Plan to Mix distributed-llm with air-llm

1. align distributed-llama to python
2. clone a airllm
3. make every model distributed to all nodes. (Shared-Storage Zero-Data Movement)
4. mix airllm + distributed-llama
5. under this target, re-organize / rewrite bottleneck to c/c++

# Distributed-AirLLM Project Tracking

## Current Phase: [2. AirLLM Integration]

### Phase 1: Python Distributed-Llama Worker [Completed - 50%]
- [x] Sub-task 1.1: Create basic Python worker structure
  - [x] NetworkClient: Socket communication with C++ root node
  - [x] ConfigReader: Read NetConfig and NodeConfig from root
  - [x] Worker: Main worker loop skeleton
  - [x] Protocol compatibility with C++ (ACK, chunked I/O)
- [x] Sub-task 1.2: Create AirLLM layer-wise components
  - [x] LayerOffsetCalculator: Calculate byte offsets for layers
  - [x] MemoryMappedWeights: Zero-copy weight loading with numpy.memmap
  - [x] LayerWiseInferenceEngine: Layer-by-layer execution framework
- [ ] Sub-task 1.3: [TODO] Implement tensor operations
  - [ ] RMS normalization
  - [ ] Matrix multiplication (support Q40, Q80, F32)
  - [ ] RoPE (Rotary Position Embedding)
  - [ ] Multi-head attention
  - [ ] SiLU/GELU activation
  - [ ] FFN (Feed-forward network)
- [ ] Sub-task 1.4: [TODO] Implement activation synchronization
  - [ ] Receive activations from root/previous node
  - [ ] Send activations to root/next node
  - [ ] Handle sync protocol (pre-sync, post-sync)
- [ ] Sub-task 1.5: [TODO] Implement weight loading protocol
  - [ ] Receive weight load commands from root
  - [ ] Load weights from memory-mapped file at specified offsets
  - [ ] Support distributed weight sharding

### Phase 2: AirLLM Integration [In Progress - 60%]
- [x] Sub-task 2.1: Parse model header for architecture
  - [x] ModelHeader dataclass with all architecture parameters
  - [x] parse_model_header() function to read binary header
  - [x] Support for LLAMA, QWEN3, QWEN3_MOE architectures
  - [x] Handle different quantization formats (F32, Q40, Q80)
- [x] Sub-task 2.2: Calculate exact weight offsets
  - [x] WeightOffsetCalculator: Calculate byte offsets for all tensors
  - [x] LayerWeightOffsets: Offsets for attention, FFN, norm weights
  - [x] Support for per-layer weight loading
  - [x] Handle different weight types (F32, Q40, Q80)
- [x] Sub-task 2.3: Integrate header parsing with layer engine
  - [x] Updated LayerWiseInferenceEngine to use parse_model_header()
  - [x] Updated MemoryMappedWeights to use WeightOffsetCalculator
  - [x] load_layer_weights() returns dict of named tensors
- [ ] Sub-task 2.4: [TODO] Implement layer caching strategy
  - [ ] LRU cache for recently used layers
  - [ ] Prefetch next layer while executing current
  - [ ] Memory pressure management
- [ ] Sub-task 2.5: [TODO] Integrate with distributed worker
  - [ ] Load only assigned layer subset per node
  - [ ] Coordinate layer distribution across nodes

### Phase 3: Zero-Data Movement Architecture [Pending]
- [ ] Sub-task 3.1: Shared storage coordination
  - [ ] Verify all nodes have same model file
  - [ ] Synchronize model checksums
- [ ] Sub-task 3.2: Optimize control signal protocol
  - [ ] Minimize metadata overhead
  - [ ] Binary protocol for offsets/indices
- [ ] Sub-task 3.3: Minimize activation transfer
  - [ ] Compression for activations
  - [ ] Quantization (F32 -> Q80)

### Phase 4: C++ Bottleneck Optimization [Pending]
- [ ] Sub-task 4.1: Profile Python implementation
  - [ ] Identify hotspots (matmul, attention, etc.)
- [ ] Sub-task 4.2: Rewrite critical ops in C++
  - [ ] Use existing nn-cpu-ops.cpp as reference
  - [ ] Create pybind11 bindings
- [ ] Sub-task 4.3: Hybrid Python/C++ worker
  - [ ] Python for control flow
  - [ ] C++ for compute kernels

## Roadmap Summary
1. **Python Alignment** [Completed - 50%]
2. **AirLLM Integration** [In Progress - 60%]
3. **Zero-Data Movement Logic** [Pending]
4. **Bottleneck C++ Rewrite** [Pending]

## Current Status
✅ Created Python worker framework with C++ protocol compatibility
✅ Created AirLLM layer-wise inference components
✅ Implemented model header parser (supports LLAMA, QWEN3, QWEN3_MOE)
✅ Implemented weight offset calculator (handles F32, Q40, Q80)
✅ Integrated header parsing with layer engine
🚧 Need to implement tensor operations (matmul, attention, etc.)
🚧 Need to implement activation synchronization
📋 Next: Implement layer caching OR implement tensor operations

