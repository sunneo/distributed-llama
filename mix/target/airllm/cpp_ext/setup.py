"""
Setup script for building C++ extension module.

Usage:
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import pybind11
import sys

# Compiler flags
extra_compile_args = ['-O3', '-std=c++11']
if sys.platform != 'win32':
    extra_compile_args.append('-march=native')

ext_modules = [
    Extension(
        'tensor_ops_cpp',
        ['tensor_ops_cpp.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name='tensor_ops_cpp',
    version='0.1.0',
    author='Distributed-Llama',
    description='Optimized C++ tensor operations',
    ext_modules=ext_modules,
    zip_safe=False,
)
