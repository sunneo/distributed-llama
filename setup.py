#!/usr/bin/env python3
"""
Setup script for Distributed-Llama with AirLLM integration.

This script packages both the distributed-llama.python worker module and
the airllm optimization module, including C++ extensions for optimal performance.

Installation:
    pip install -e .                    # Development mode
    pip install .                       # Regular installation
    
Build C++ extensions only:
    python setup.py build_ext --inplace

Environment variables:
    SKIP_CPP_BUILD=1    Skip building C++ extensions
    DEBUG=1             Build with debug symbols
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop
import os
import sys
import subprocess
import platform

# Package metadata
VERSION = "0.1.0"
DESCRIPTION = "Distributed LLM Inference with AirLLM optimization"
AUTHOR = "Distributed-Llama Contributors"
LICENSE = "MIT"

# Paths
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MIX_TARGET = os.path.join(REPO_ROOT, "mix", "target")
AIRLLM_PATH = os.path.join(MIX_TARGET, "airllm")
# Note: The source directory is named 'distributed-llama.python' (with dash)
# but we use 'distributed_llama_python' (with underscores) as the Python package name
# via a symlink to comply with Python naming conventions
DLLAMA_PY_SOURCE_DIR = "distributed-llama.python"  # Original source directory name
DLLAMA_PY_PACKAGE_DIR = "distributed_llama_python"  # Python package name (via symlink)
DLLAMA_PY_PATH = os.path.join(MIX_TARGET, DLLAMA_PY_SOURCE_DIR)
CPP_EXT_PATH = os.path.join(AIRLLM_PATH, "cpp_ext")

def read_requirements(filename):
    """Read requirements from a file."""
    req_path = os.path.join(DLLAMA_PY_PATH, filename)
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

def get_long_description():
    """Get long description from README."""
    readme_path = os.path.join(REPO_ROOT, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return DESCRIPTION

class BuildCppExtensions(_build_ext):
    """Custom build command that builds C++ extensions using the existing setup.py."""
    
    def run(self):
        """Build C++ extensions from airllm/cpp_ext."""
        if os.environ.get('SKIP_CPP_BUILD') == '1':
            print("Skipping C++ extension build (SKIP_CPP_BUILD=1)")
            return
        
        print("\n" + "="*70)
        print("Building optimized C++ extensions for tensor operations...")
        print("="*70 + "\n")
        
        # Check if cpp_ext/setup.py exists
        cpp_setup_py = os.path.join(CPP_EXT_PATH, "setup.py")
        if not os.path.exists(cpp_setup_py):
            print(f"Warning: C++ extension setup.py not found at {cpp_setup_py}")
            print("Skipping C++ extension build")
            return
        
        # Build the C++ extension using its own setup.py
        try:
            # First detect capabilities
            print("\nDetecting CPU capabilities...")
            result = subprocess.run(
                [sys.executable, "setup.py", "test_capabilities"],
                cwd=CPP_EXT_PATH,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(result.stdout)
            
            # Build the extension
            print("\nBuilding C++ extension...")
            build_cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
            result = subprocess.run(
                build_cmd,
                cwd=CPP_EXT_PATH,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            print("\n✓ C++ extensions built successfully!")
            print("  Location: mix/target/airllm/cpp_ext/")
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Warning: Failed to build C++ extensions: {e}")
            # Note: stdout and stderr are always available when capture_output=True
            print("  Standard output:", e.stdout)
            print("  Standard error:", e.stderr)
            print("\n  The package will still be installed, but without C++ optimizations.")
            print("  You can manually build the C++ extensions later by running:")
            print(f"    cd {CPP_EXT_PATH}")
            print("    python setup.py build_ext --inplace")
        except Exception as e:
            print(f"\n✗ Warning: Unexpected error building C++ extensions: {e}")

class CustomInstall(_install):
    """Custom install command that builds C++ extensions."""
    
    def run(self):
        # Build C++ extensions first
        self.run_command('build_ext')
        # Then run standard install
        _install.run(self)

class CustomDevelop(_develop):
    """Custom develop command that builds C++ extensions."""
    
    def run(self):
        # Build C++ extensions first
        self.run_command('build_ext')
        # Then run standard develop
        _develop.run(self)

def get_package_data():
    """Get package data including C++ extensions and other files."""
    package_data = {
        'airllm': [
            '*.md',
            'cpp_ext/*.cpp',
            'cpp_ext/*.cu',
            'cpp_ext/*.py',
            'cpp_ext/*.so',
            'cpp_ext/*.pyd',
            'cpp_ext/*.dll',
            'cpp_ext/*.json',
            'examples/*.py',
        ],
        'distributed_llama_python': [
            '*.txt',
            '*.md',
        ],
    }
    return package_data

# Main setup configuration
setup(
    name="distributed-llama",
    version=VERSION,
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    license=LICENSE,
    url="https://github.com/b4rtaz/distributed-llama",
    
    # Package discovery
    packages=[
        'airllm',
        'distributed_llama_python',
    ],
    package_dir={
        'airllm': 'mix/target/airllm',
        'distributed_llama_python': 'mix/target/distributed_llama_python',
    },
    package_data=get_package_data(),
    include_package_data=True,
    
    # Dependencies
    install_requires=read_requirements('requirements.txt'),
    python_requires='>=3.8',
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'dllama-worker=distributed_llama_python.worker:main',
        ],
    },
    
    # Custom build commands
    cmdclass={
        'build_ext': BuildCppExtensions,
        'install': CustomInstall,
        'develop': CustomDevelop,
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    # Zip safety
    zip_safe=False,
)
