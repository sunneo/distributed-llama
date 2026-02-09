#!/usr/bin/env python3
"""
Test script to verify the installation works correctly.
"""
import sys
import os
import subprocess

def test_imports():
    """Test that modules can be imported."""
    print("=" * 70)
    print("Testing module imports...")
    print("=" * 70)
    
    try:
        print("\n1. Testing airllm import...")
        import airllm
        print(f"   ✓ airllm version: {airllm.__version__}")
        print(f"   ✓ airllm location: {airllm.__file__}")
        
        print("\n2. Testing airllm submodules...")
        from airllm import ModelHeader, LayerWiseInferenceEngine, tensor_ops
        print("   ✓ ModelHeader imported")
        print("   ✓ LayerWiseInferenceEngine imported")
        print("   ✓ tensor_ops imported")
        
        print("\n3. Testing distributed_llama_python import...")
        import distributed_llama_python
        print(f"   ✓ distributed_llama_python version: {distributed_llama_python.__version__}")
        print(f"   ✓ distributed_llama_python location: {distributed_llama_python.__file__}")
        
        print("\n4. Testing distributed_llama_python submodules...")
        from distributed_llama_python import Worker, NetworkClient, NetConfig
        print("   ✓ Worker imported")
        print("   ✓ NetworkClient imported")
        print("   ✓ NetConfig imported")
        
        print("\n✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False

def test_cpp_extensions():
    """Test if C++ extensions are available."""
    print("\n" + "=" * 70)
    print("Testing C++ extensions...")
    print("=" * 70)
    
    try:
        from airllm import tensor_ops
        
        # Check if C++ backend is available
        if hasattr(tensor_ops, 'detect_backend'):
            backend = tensor_ops.detect_backend()
            print(f"\n   Current backend: {backend}")
            
            if backend == 'cpp':
                print("   ✓ C++ extension is available and active!")
                return True
            elif backend == 'python':
                print("   ℹ Using pure Python backend (C++ not built)")
                return True
            else:
                print(f"   ℹ Using {backend} backend")
                return True
        else:
            print("   ℹ Backend detection not available in this version")
            return True
            
    except Exception as e:
        print(f"   ✗ Error checking C++ extensions: {e}")
        return False

def test_entry_points():
    """Test if entry points are installed."""
    print("\n" + "=" * 70)
    print("Testing entry points...")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            ['dllama-worker', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("\n   ✓ dllama-worker command is available")
            print("   Sample output:")
            for line in result.stdout.split('\n')[:5]:
                print(f"     {line}")
            return True
        else:
            print("\n   ✗ dllama-worker command failed")
            return False
            
    except FileNotFoundError:
        print("\n   ✗ dllama-worker command not found")
        print("   This is expected if not installed via pip")
        return False
    except Exception as e:
        print(f"\n   ✗ Error testing entry point: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DISTRIBUTED-LLAMA INSTALLATION TEST")
    print("=" * 70)
    
    results = []
    
    # Test imports
    results.append(("Module imports", test_imports()))
    
    # Test C++ extensions
    results.append(("C++ extensions", test_cpp_extensions()))
    
    # Test entry points
    results.append(("Entry points", test_entry_points()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
