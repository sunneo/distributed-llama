"""
AirLLM - Layer-wise inference for large language models

Enables running large models (30B+) on consumer hardware by:
1. Loading only 1-2 layers into RAM at a time
2. Swapping layers from disk as needed
3. Using memory-mapped I/O for zero-copy access
"""

__version__ = "0.1.0"

from .layer_engine import (
    LayerOffsetCalculator,
    MemoryMappedWeights,
    LayerWiseInferenceEngine
)

__all__ = [
    'LayerOffsetCalculator',
    'MemoryMappedWeights', 
    'LayerWiseInferenceEngine'
]
