"""
Shared storage coordination for Zero-Data Movement Architecture.

This module implements utilities for verifying that all nodes have access
to the same model file, enabling the zero-data movement paradigm where
only activations are transferred over the network.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any


class StorageCoordinator:
    """
    Coordinates shared storage access across distributed nodes.
    
    Ensures all nodes have the same model file by verifying:
    - File existence and accessibility
    - File size consistency
    - Content checksums (MD5 hash)
    """
    
    def __init__(self, model_path: str):
        """
        Initialize storage coordinator.
        
        Args:
            model_path: Path to model file
        """
        self.model_path = Path(model_path)
        self._cached_checksum: Optional[str] = None
        self._cached_size: Optional[int] = None
    
    def verify_file_exists(self) -> bool:
        """
        Verify model file exists and is accessible.
        
        Returns:
            True if file exists and is readable
        """
        if not self.model_path.exists():
            print(f"ERROR: Model file not found: {self.model_path}")
            return False
        
        if not self.model_path.is_file():
            print(f"ERROR: Path is not a file: {self.model_path}")
            return False
        
        if not os.access(self.model_path, os.R_OK):
            print(f"ERROR: Model file is not readable: {self.model_path}")
            return False
        
        return True
    
    def get_file_size(self) -> int:
        """
        Get model file size in bytes.
        
        Returns:
            File size in bytes
        """
        if self._cached_size is None:
            self._cached_size = self.model_path.stat().st_size
        return self._cached_size
    
    def compute_checksum(self, chunk_size: int = 8192) -> str:
        """
        Compute MD5 checksum of model file.
        
        For large files, computes in chunks to avoid loading entire file.
        
        Args:
            chunk_size: Size of chunks to read (default 8KB)
            
        Returns:
            MD5 checksum as hex string
        """
        if self._cached_checksum is not None:
            return self._cached_checksum
        
        md5 = hashlib.md5()
        
        with open(self.model_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5.update(chunk)
        
        self._cached_checksum = md5.hexdigest()
        return self._cached_checksum
    
    def compute_fast_checksum(self, sample_size: int = 1024 * 1024) -> str:
        """
        Compute fast checksum by sampling file (beginning, middle, end).
        
        For very large model files (30GB+), computing full checksum is slow.
        This samples beginning, middle, and end of file for quick verification.
        
        Args:
            sample_size: Size of each sample in bytes (default 1MB)
            
        Returns:
            MD5 checksum of sampled content
        """
        file_size = self.get_file_size()
        md5 = hashlib.md5()
        
        with open(self.model_path, 'rb') as f:
            # Sample beginning
            chunk = f.read(min(sample_size, file_size))
            md5.update(chunk)
            
            # Sample middle (if file is large enough)
            if file_size > sample_size * 3:
                f.seek(file_size // 2)
                chunk = f.read(sample_size)
                md5.update(chunk)
            
            # Sample end (if file is large enough)
            if file_size > sample_size * 2:
                f.seek(max(0, file_size - sample_size))
                chunk = f.read(sample_size)
                md5.update(chunk)
        
        return md5.hexdigest()
    
    def get_file_info(self, fast_checksum: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            fast_checksum: Use fast sampling-based checksum (default True)
            
        Returns:
            Dictionary with file metadata
        """
        if not self.verify_file_exists():
            return {
                'exists': False,
                'path': str(self.model_path),
                'error': 'File not found or not accessible'
            }
        
        file_size = self.get_file_size()
        checksum = self.compute_fast_checksum() if fast_checksum else self.compute_checksum()
        
        return {
            'exists': True,
            'path': str(self.model_path),
            'size': file_size,
            'size_gb': file_size / (1024 ** 3),
            'checksum': checksum,
            'checksum_type': 'fast_md5' if fast_checksum else 'full_md5'
        }
    
    def verify_against(self, other_info: Dict[str, Any]) -> bool:
        """
        Verify this file matches another node's file.
        
        Args:
            other_info: File info from another node (from get_file_info())
            
        Returns:
            True if files match
        """
        local_info = self.get_file_info(fast_checksum=True)
        
        if not local_info['exists']:
            print(f"ERROR: Local file does not exist")
            return False
        
        if not other_info.get('exists', False):
            print(f"ERROR: Remote file does not exist")
            return False
        
        # Check size match
        if local_info['size'] != other_info['size']:
            print(f"ERROR: File size mismatch: local={local_info['size']}, "
                  f"remote={other_info['size']}")
            return False
        
        # Check checksum match
        if local_info['checksum'] != other_info['checksum']:
            print(f"ERROR: File checksum mismatch: local={local_info['checksum']}, "
                  f"remote={other_info['checksum']}")
            return False
        
        print(f"✓ File verification passed: {self.model_path.name}")
        print(f"  Size: {local_info['size_gb']:.2f} GB")
        print(f"  Checksum: {local_info['checksum']}")
        
        return True


def verify_shared_storage(model_path: str, nodes_info: list) -> bool:
    """
    Verify all nodes have the same model file.
    
    Args:
        model_path: Local path to model file
        nodes_info: List of file info dicts from other nodes
        
    Returns:
        True if all nodes have matching files
    """
    coordinator = StorageCoordinator(model_path)
    local_info = coordinator.get_file_info(fast_checksum=True)
    
    if not local_info['exists']:
        print(f"ERROR: Local model file not accessible")
        return False
    
    print(f"Verifying shared storage across {len(nodes_info)} nodes...")
    print(f"Local file: {local_info['path']} ({local_info['size_gb']:.2f} GB)")
    
    all_match = True
    for i, node_info in enumerate(nodes_info):
        if not coordinator.verify_against(node_info):
            print(f"ERROR: Node {i} file does not match")
            all_match = False
    
    if all_match:
        print(f"✓ All nodes verified: shared storage OK")
    
    return all_match
