#!/usr/bin/env python3
"""
Example: Backend Configuration

This example demonstrates how to configure backend preferences
using the backend.json configuration file.
"""

import sys
import os
import subprocess

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def example_view_config():
    """Example 1: View current backend configuration."""
    print_section("Example 1: View Current Configuration")
    
    result = subprocess.run(
        [sys.executable, 'setup.py', 'configure_backend', '--show'],
        capture_output=True,
        text=True
    )
    print(result.stdout)


def example_set_backend():
    """Example 2: Set preferred backend."""
    print_section("Example 2: Set Preferred Backend to C++")
    
    result = subprocess.run(
        [sys.executable, 'setup.py', 'configure_backend', '--backend=cpp'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    # Verify it worked
    print("\nVerifying configuration:")
    import tensor_ops
    # Clear cache to reload config
    tensor_ops._backend = None
    tensor_ops._backend_module = None
    tensor_ops._backend_config = None
    
    print(f"Active backend: {tensor_ops.get_backend()}")
    print(f"Config: {tensor_ops.get_backend_info()['config']['preferred_backend']}")


def example_set_priority():
    """Example 3: Set backend priority."""
    print_section("Example 3: Set Custom Backend Priority")
    
    result = subprocess.run(
        [sys.executable, 'setup.py', 'configure_backend', 
         '--backend=auto', '--priority=cpp,cuda,opencl,python'],
        capture_output=True,
        text=True
    )
    print(result.stdout)


def example_runtime_check():
    """Example 4: Check configuration at runtime."""
    print_section("Example 4: Runtime Configuration Check")
    
    import tensor_ops
    
    info = tensor_ops.get_backend_info()
    
    print(f"Active backend: {info['backend']}")
    print(f"Available backends: {', '.join(info['available_backends'])}")
    print(f"\nConfiguration:")
    print(f"  Preferred: {info['config']['preferred_backend']}")
    print(f"  Priority: {', '.join(info['config']['backend_priority'])}")
    print(f"  Force: {info['config'].get('force_backend', 'None')}")


def example_restore_auto():
    """Example 5: Restore automatic detection."""
    print_section("Example 5: Restore Automatic Detection")
    
    result = subprocess.run(
        [sys.executable, 'setup.py', 'configure_backend', '--backend=auto'],
        capture_output=True,
        text=True
    )
    print(result.stdout)


def main():
    """Run all examples."""
    print("=" * 70)
    print("Backend Configuration Examples")
    print("=" * 70)
    
    # Change to cpp_ext directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        example_view_config()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example_set_backend()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example_set_priority()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example_runtime_check()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    try:
        example_restore_auto()
    except Exception as e:
        print(f"\nExample 5 failed: {e}")
    
    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Read BACKEND_CONFIG.md for detailed configuration options")
    print("  - Use 'python setup.py configure_backend --help' for command help")
    print("  - Edit backend.json directly for manual configuration")


if __name__ == '__main__':
    main()
