"""
Layer caching strategy for AirLLM.

This module implements LRU (Least Recently Used) cache for layer weights
with prefetching and memory pressure management.
"""

import numpy as np
from typing import Dict, Optional, List, Set
from collections import OrderedDict
import psutil


class LayerCache:
    """
    LRU cache for layer weights with prefetching.
    
    Caches recently used layers in RAM while managing memory pressure.
    Supports prefetching of next layer during current layer execution.
    """
    
    def __init__(self, max_layers: int = 2, max_memory_gb: float = 4.0):
        """
        Initialize layer cache.
        
        Args:
            max_layers: Maximum number of layers to cache
            max_memory_gb: Maximum memory usage in GB
        """
        self.max_layers = max_layers
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        
        # LRU cache: layer_id -> weights dict
        self.cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()
        
        # Track cache size in bytes
        self.cache_size_bytes = 0
        
        # Prefetch queue
        self.prefetch_queue: List[int] = []
        
        # Currently loading layers (to avoid duplicate loads)
        self.loading: Set[int] = set()
    
    def get_memory_usage(self) -> float:
        """
        Get current system memory usage.
        
        Returns:
            Memory usage as fraction (0.0 to 1.0)
        """
        memory = psutil.virtual_memory()
        return memory.percent / 100.0
    
    def should_evict(self) -> bool:
        """
        Check if cache should evict entries.
        
        Returns:
            True if eviction needed due to memory pressure
        """
        # Evict if cache exceeds max layers
        if len(self.cache) >= self.max_layers:
            return True
        
        # Evict if cache size exceeds max memory
        if self.cache_size_bytes >= self.max_memory_bytes:
            return True
        
        # Evict if system memory pressure is high (>90%)
        if self.get_memory_usage() > 0.9:
            return True
        
        return False
    
    def evict_lru(self) -> None:
        """Evict least recently used layer from cache."""
        if not self.cache:
            return
        
        # Pop oldest (first) item from OrderedDict
        layer_id, weights = self.cache.popitem(last=False)
        
        # Update cache size
        layer_size = sum(w.nbytes for w in weights.values())
        self.cache_size_bytes -= layer_size
        
        print(f"Evicted layer {layer_id} from cache ({layer_size / 1024 / 1024:.2f} MB)")
    
    def get(self, layer_id: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Get layer weights from cache.
        
        Updates LRU order on access.
        
        Args:
            layer_id: Layer index
            
        Returns:
            Layer weights dict if cached, None otherwise
        """
        if layer_id not in self.cache:
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(layer_id)
        
        return self.cache[layer_id]
    
    def put(self, layer_id: int, weights: Dict[str, np.ndarray]) -> None:
        """
        Put layer weights into cache.
        
        Args:
            layer_id: Layer index
            weights: Dictionary of weight tensors
        """
        # Calculate layer size
        layer_size = sum(w.nbytes for w in weights.values())
        
        # Evict if needed
        while self.should_evict() and len(self.cache) > 0:
            self.evict_lru()
        
        # Add to cache
        self.cache[layer_id] = weights
        self.cache_size_bytes += layer_size
        
        # Mark as not loading
        self.loading.discard(layer_id)
        
        print(f"Cached layer {layer_id} ({layer_size / 1024 / 1024:.2f} MB, "
              f"total: {self.cache_size_bytes / 1024 / 1024:.2f} MB)")
    
    def prefetch(self, layer_id: int) -> None:
        """
        Add layer to prefetch queue.
        
        Args:
            layer_id: Layer index to prefetch
        """
        if layer_id not in self.cache and layer_id not in self.loading:
            if layer_id not in self.prefetch_queue:
                self.prefetch_queue.append(layer_id)
    
    def get_next_prefetch(self) -> Optional[int]:
        """
        Get next layer to prefetch.
        
        Returns:
            Layer ID to prefetch, or None if queue empty
        """
        if not self.prefetch_queue:
            return None
        
        # Pop from front of queue
        layer_id = self.prefetch_queue.pop(0)
        
        # Skip if already cached or loading
        while (layer_id in self.cache or layer_id in self.loading) and self.prefetch_queue:
            layer_id = self.prefetch_queue.pop(0)
        
        if layer_id not in self.cache and layer_id not in self.loading:
            self.loading.add(layer_id)
            return layer_id
        
        return None
    
    def mark_loading(self, layer_id: int) -> None:
        """
        Mark layer as currently loading.
        
        Args:
            layer_id: Layer index
        """
        self.loading.add(layer_id)
    
    def clear(self) -> None:
        """Clear all cached layers."""
        self.cache.clear()
        self.cache_size_bytes = 0
        self.prefetch_queue.clear()
        self.loading.clear()
        print("Cleared layer cache")
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'cached_layers': list(self.cache.keys()),
            'n_cached': len(self.cache),
            'cache_size_mb': self.cache_size_bytes / 1024 / 1024,
            'prefetch_queue': self.prefetch_queue.copy(),
            'loading': list(self.loading),
            'memory_usage': self.get_memory_usage()
        }
