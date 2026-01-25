"""
Setup script for building C++ extension module with capability detection.

Usage:
    python setup.py build_ext --inplace
    python setup.py test_capabilities  # Test what optimizations are available
"""

from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
import pybind11
import subprocess
import sys
import os
import platform
import tempfile
import shutil


class CapabilityDetector:
    """Detect available hardware capabilities and compiler support."""
    
    def __init__(self):
        self.capabilities = {
            'openmp': False,
            'avx': False,
            'avx2': False,
            'avx512': False,
            'fma': False,
            'neon': False,
            'cuda': False,
            'opencl': False,
            'vulkan': False,
        }
        self.compiler = self._detect_compiler()
        self.runtime_cpu_flags = self._get_cpu_flags()
        
    def _detect_compiler(self):
        """Detect which C++ compiler is available."""
        for compiler in ['g++', 'clang++', 'c++']:
            try:
                result = subprocess.run([compiler, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return compiler
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return None
    
    def _get_cpu_flags(self):
        """Get CPU flags from /proc/cpuinfo on Linux."""
        flags = set()
        try:
            if sys.platform.startswith('linux'):
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('flags'):
                            flags_str = line.split(':')[1].strip()
                            flags = set(flags_str.split())
                            break
        except:
            pass
        return flags
    
    def _test_compile_and_run(self, code, flags):
        """
        Test if code compiles and runs successfully.
        
        Note: This runs compiled test programs to verify runtime CPU support.
        The test programs are simple (just initialize SIMD registers) and
        execute in an isolated temporary directory. This is necessary to
        distinguish between compile-time support (compiler can generate code)
        and runtime support (CPU can execute the instructions).
        """
        if not self.compiler:
            return False
            
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, 'test.cpp')
            out_file = os.path.join(tmpdir, 'test' + ('.exe' if sys.platform == 'win32' else ''))
            
            with open(src_file, 'w') as f:
                f.write(code)
            
            # Compile
            cmd = [self.compiler, src_file, '-o', out_file] + flags
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                if result.returncode != 0:
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
            
            # Try to run
            try:
                result = subprocess.run([out_file], capture_output=True, timeout=5)
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
    
    def _test_compile(self, code, flags):
        """Test if code compiles with given flags."""
        if not self.compiler:
            return False
            
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, 'test.cpp')
            out_file = os.path.join(tmpdir, 'test.o')
            
            with open(src_file, 'w') as f:
                f.write(code)
            
            cmd = [self.compiler, '-c', src_file, '-o', out_file] + flags
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
    
    def detect_openmp(self):
        """Detect OpenMP support."""
        code = """
        #include <omp.h>
        int main() {
            #pragma omp parallel
            { int tid = omp_get_thread_num(); }
            return 0;
        }
        """
        flags = ['-fopenmp'] if sys.platform != 'win32' else ['/openmp']
        self.capabilities['openmp'] = self._test_compile(code, flags)
        return self.capabilities['openmp']
    
    def detect_avx(self):
        """Detect AVX support."""
        # First check CPU flags
        if 'avx' not in self.runtime_cpu_flags and sys.platform.startswith('linux'):
            self.capabilities['avx'] = False
            return False
        
        code = """
        #include <immintrin.h>
        int main() {
            __m256 a = _mm256_setzero_ps();
            return 0;
        }
        """
        self.capabilities['avx'] = self._test_compile_and_run(code, ['-mavx'])
        return self.capabilities['avx']
    
    def detect_avx2(self):
        """Detect AVX2 support."""
        # First check CPU flags
        if 'avx2' not in self.runtime_cpu_flags and sys.platform.startswith('linux'):
            self.capabilities['avx2'] = False
            return False
        
        code = """
        #include <immintrin.h>
        int main() {
            __m256i a = _mm256_setzero_si256();
            return 0;
        }
        """
        self.capabilities['avx2'] = self._test_compile_and_run(code, ['-mavx2'])
        return self.capabilities['avx2']
    
    def detect_avx512(self):
        """Detect AVX-512 support."""
        # First check CPU flags
        if 'avx512f' not in self.runtime_cpu_flags and sys.platform.startswith('linux'):
            self.capabilities['avx512'] = False
            return False
        
        code = """
        #include <immintrin.h>
        int main() {
            __m512 a = _mm512_setzero_ps();
            return 0;
        }
        """
        self.capabilities['avx512'] = self._test_compile_and_run(code, ['-mavx512f'])
        return self.capabilities['avx512']
    
    def detect_fma(self):
        """Detect FMA support."""
        # First check CPU flags
        if 'fma' not in self.runtime_cpu_flags and sys.platform.startswith('linux'):
            self.capabilities['fma'] = False
            return False
        
        code = """
        #include <immintrin.h>
        int main() {
            __m256 a = _mm256_setzero_ps();
            __m256 b = _mm256_setzero_ps();
            __m256 c = _mm256_fmadd_ps(a, b, a);
            return 0;
        }
        """
        self.capabilities['fma'] = self._test_compile_and_run(code, ['-mfma', '-mavx2'])
        return self.capabilities['fma']
    
    def detect_neon(self):
        """Detect ARM NEON support."""
        code = """
        #include <arm_neon.h>
        int main() {
            float32x4_t a = vdupq_n_f32(0.0f);
            return 0;
        }
        """
        flags = ['-march=armv8-a+simd'] if 'arm' in platform.machine().lower() or 'aarch64' in platform.machine().lower() else []
        self.capabilities['neon'] = self._test_compile(code, flags)
        return self.capabilities['neon']
    
    def detect_cuda(self):
        """Detect CUDA availability."""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, timeout=5)
            self.capabilities['cuda'] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.capabilities['cuda'] = False
        return self.capabilities['cuda']
    
    def detect_opencl(self):
        """Detect OpenCL availability."""
        code = """
        #ifdef __APPLE__
        #include <OpenCL/opencl.h>
        #else
        #include <CL/cl.h>
        #endif
        int main() {
            cl_uint num_platforms;
            clGetPlatformIDs(0, NULL, &num_platforms);
            return 0;
        }
        """
        flags = ['-lOpenCL'] if sys.platform != 'win32' else []
        self.capabilities['opencl'] = self._test_compile(code, flags)
        return self.capabilities['opencl']
    
    def detect_vulkan(self):
        """Detect Vulkan availability."""
        code = """
        #include <vulkan/vulkan.h>
        int main() {
            VkInstance instance;
            return 0;
        }
        """
        flags = ['-lvulkan'] if sys.platform != 'win32' else []
        self.capabilities['vulkan'] = self._test_compile(code, flags)
        return self.capabilities['vulkan']
    
    def detect_all(self):
        """Run all capability detections."""
        print("Detecting CPU and compiler capabilities...\n")
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Compiler: {self.compiler or 'None found'}\n")
        
        if not self.compiler:
            print("ERROR: No C++ compiler found!")
            return self.capabilities
        
        tests = [
            ('OpenMP', self.detect_openmp),
            ('AVX', self.detect_avx),
            ('AVX2', self.detect_avx2),
            ('AVX-512', self.detect_avx512),
            ('FMA', self.detect_fma),
            ('ARM NEON', self.detect_neon),
            ('CUDA', self.detect_cuda),
            ('OpenCL', self.detect_opencl),
            ('Vulkan', self.detect_vulkan),
        ]
        
        for name, test_func in tests:
            result = test_func()
            status = "✓ Available" if result else "✗ Not available"
            print(f"{name:15} {status}")
        
        print("\nRecommended build configuration:")
        flags = self.get_recommended_flags()
        print(f"  Compile flags: {' '.join(flags)}")
        
        return self.capabilities
    
    def get_recommended_flags(self):
        """Get recommended compiler flags based on detected capabilities."""
        flags = ['-O3', '-std=c++11']
        
        if sys.platform != 'win32':
            # Use native optimizations if possible
            flags.append('-march=native')
            
            if self.capabilities['openmp']:
                flags.append('-fopenmp')
            
            # SIMD flags (ordered by preference)
            if self.capabilities['avx512']:
                flags.extend(['-mavx512f', '-mavx512dq'])
            elif self.capabilities['avx2'] and self.capabilities['fma']:
                flags.extend(['-mavx2', '-mfma'])
            elif self.capabilities['avx']:
                flags.append('-mavx')
        else:
            # Windows-specific flags
            if self.capabilities['openmp']:
                flags.append('/openmp')
            if self.capabilities['avx2']:
                flags.append('/arch:AVX2')
            elif self.capabilities['avx']:
                flags.append('/arch:AVX')
        
        return flags


class TestCapabilitiesCommand(Command):
    """Custom command to test available capabilities."""
    description = 'Test available CPU and compiler capabilities'
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        detector = CapabilityDetector()
        capabilities = detector.detect_all()
        
        print("\n" + "="*60)
        print("Capability Detection Complete")
        print("="*60)
        print("\nTo build with detected optimizations:")
        print("  python setup.py build_ext --inplace")
        
        if capabilities['cuda']:
            print("\nOptional CUDA backend detected:")
            print("  To build CUDA backend (experimental):")
            print("    python setup.py build_cuda")
        
        if capabilities['opencl']:
            print("\nOptional OpenCL backend detected:")
            print("  To build OpenCL backend (experimental):")
            print("    python setup.py build_opencl")
        
        if capabilities['vulkan']:
            print("\nOptional Vulkan backend detected:")
            print("  To build Vulkan backend (experimental):")
            print("    python setup.py build_vulkan")


class BuildCudaCommand(Command):
    """Build CUDA backend extension."""
    description = 'Build CUDA backend (requires CUDA toolkit)'
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        detector = CapabilityDetector()
        if not detector.detect_cuda():
            print("ERROR: CUDA not available on this system")
            print("Please install CUDA toolkit from https://developer.nvidia.com/cuda-downloads")
            return
        
        print("Building CUDA backend...")
        # Build with nvcc
        cmd = [
            'nvcc', '-O3', '-shared', '-Xcompiler', '-fPIC',
            '--compiler-options', '-I' + pybind11.get_include(),
            '-o', 'tensor_ops_cuda' + self._get_extension_suffix(),
            'tensor_ops_cuda.cu'
        ]
        
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or '.')
        if result.returncode == 0:
            print("✓ CUDA backend built successfully")
        else:
            print("✗ CUDA backend build failed")
    
    def _get_extension_suffix(self):
        import sysconfig
        return sysconfig.get_config_var('EXT_SUFFIX') or '.so'


class BuildOpenCLCommand(Command):
    """Build OpenCL backend extension."""
    description = 'Build OpenCL backend (requires OpenCL)'
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        detector = CapabilityDetector()
        if not detector.detect_opencl():
            print("ERROR: OpenCL not available on this system")
            print("Please install OpenCL drivers for your GPU")
            return
        
        print("Building OpenCL backend...")
        print("Note: OpenCL backend is experimental and requires manual build")
        print("\nBuild command:")
        print("  g++ -O3 -shared -std=c++11 -fPIC \\")
        print("      $(python3 -m pybind11 --includes) \\")
        print("      -lOpenCL \\")
        print("      -o tensor_ops_opencl$(python3-config --extension-suffix) \\")
        print("      tensor_ops_opencl.cpp")


class CustomBuildExt(build_ext):
    """Custom build_ext command that auto-detects capabilities."""
    
    def build_extensions(self):
        # Detect capabilities
        detector = CapabilityDetector()
        detector.detect_all()
        
        # Update compile flags for all extensions
        for ext in self.extensions:
            flags = detector.get_recommended_flags()
            ext.extra_compile_args = flags
            
            # Add linker flags for OpenMP
            if detector.capabilities['openmp']:
                if sys.platform != 'win32':
                    ext.extra_link_args = ['-fopenmp']
                else:
                    ext.extra_link_args = []
            else:
                ext.extra_link_args = []
            
            # Add preprocessor defines for detected features
            if detector.capabilities['openmp']:
                ext.define_macros.append(('USE_OPENMP', '1'))
            if detector.capabilities['avx512']:
                ext.define_macros.append(('USE_AVX512', '1'))
            if detector.capabilities['avx2']:
                ext.define_macros.append(('USE_AVX2', '1'))
            if detector.capabilities['avx']:
                ext.define_macros.append(('USE_AVX', '1'))
            if detector.capabilities['fma']:
                ext.define_macros.append(('USE_FMA', '1'))
            if detector.capabilities['neon']:
                ext.define_macros.append(('USE_NEON', '1'))
            
            print(f"\nBuilding {ext.name} with optimizations:")
            print(f"  Flags: {' '.join(flags)}")
            print(f"  Defines: {ext.define_macros}")
            print(f"  Link flags: {ext.extra_link_args}")
        
        # Call parent build
        build_ext.build_extensions(self)


# Setup extension module
ext_modules = [
    Extension(
        'tensor_ops_cpp',
        ['tensor_ops_cpp.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++11'],  # Will be overridden by CustomBuildExt
        define_macros=[],
    ),
]

setup(
    name='tensor_ops_cpp',
    version='0.2.0',
    author='Distributed-Llama',
    description='Optimized C++ tensor operations with multi-level CPU optimizations',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CustomBuildExt,
        'test_capabilities': TestCapabilitiesCommand,
        'build_cuda': BuildCudaCommand,
        'build_opencl': BuildOpenCLCommand,
    },
    zip_safe=False,
)
