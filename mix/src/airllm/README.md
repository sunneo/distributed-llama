# AirLLM - Layer-wise Inference

Layer-wise inference engine for running large language models on consumer hardware.

## Concept

AirLLM enables running models that don't fit in RAM by:

1. **Layer-wise Loading**: Only load 1-2 transformer layers at a time
2. **Disk Swapping**: Swap layers from SSD as needed
3. **Zero-copy I/O**: Use memory-mapped files (numpy.memmap) to avoid unnecessary data copying

## Components

### LayerOffsetCalculator

Calculates byte offsets for each layer in the model file:

```python
calc = LayerOffsetCalculator('model.m')
calc.calculate_offsets(n_layers=32, dim=4096, hidden_dim=11008)
offset, size = calc.get_layer_offset(layer_id=5)
```

### MemoryMappedWeights

Zero-copy weight loader using numpy.memmap:

```python
loader = MemoryMappedWeights('model.m')
loader.open()
weights = loader.load_layer_weights(offset, size)  # No copy, just a view
loader.close()
```

### LayerWiseInferenceEngine

Main inference engine that orchestrates layer-wise execution:

```python
engine = LayerWiseInferenceEngine('model.m')
engine.initialize(n_layers=32, dim=4096, hidden_dim=11008)
output = engine.forward(input_tokens, n_layers=32)
engine.cleanup()
```

## TODO

- [ ] Parse model header to get exact architecture
- [ ] Implement tensor shape calculation for each layer
- [ ] Support different quantization formats (Q40, Q80, F32)
- [ ] Implement actual layer operations (attention, FFN)
- [ ] Add caching for recently used layers
- [ ] Optimize disk I/O patterns

## Integration with Distributed-Llama

When combined with Distributed-Llama:

1. Each node has the full model on SSD
2. Nodes only transmit activations (small), not weights (large)
3. Each node loads only its assigned layer subset
4. "Shared-Storage Zero-Data Movement" architecture

```
Node 0: Layers 0-7   (loads from local SSD)
Node 1: Layers 8-15  (loads from local SSD)  
Node 2: Layers 16-23 (loads from local SSD)
Node 3: Layers 24-31 (loads from local SSD)

Network traffic = activations only (~few KB per token)
```
