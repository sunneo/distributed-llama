"""
AirLLM - Layer-wise inference for large language models

Enables running large models (30B+) on consumer hardware by:
1. Loading only 1-2 layers into RAM at a time
2. Swapping layers from disk as needed
3. Using memory-mapped I/O for zero-copy access
"""

__version__ = "0.1.0"

from .layer_engine import (
    MemoryMappedWeights,
    LayerWiseInferenceEngine
)
from .model_header import (
    ModelHeader,
    parse_model_header,
    print_model_header,
    ArchType,
    FloatType,
    HiddenAct,
    RopeType
)
from .weight_offsets import (
    WeightOffsetCalculator,
    LayerWeightOffsets
)
from .layer_cache import LayerCache
from . import tensor_ops

__all__ = [
    'MemoryMappedWeights', 
    'LayerWiseInferenceEngine',
    'ModelHeader',
    'parse_model_header',
    'print_model_header',
    'ArchType',
    'FloatType',
    'HiddenAct',
    'RopeType',
    'WeightOffsetCalculator',
    'LayerWeightOffsets',
    'LayerCache',
    'tensor_ops'
]
