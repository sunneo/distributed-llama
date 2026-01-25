/*
 * C++ Extension Module for Distributed-Llama Python Worker
 *
 * This module provides Python bindings to optimized C++ implementations
 * of critical tensor operations with multi-level CPU optimizations:
 * - Matrix multiplication (using BLAS)
 * - RMS normalization (with SIMD)
 * - Activation functions (SiLU, GELU)
 *
 * Optimization levels (auto-detected):
 * - AVX-512: Intel/AMD CPUs with AVX-512 support
 * - AVX2+FMA: Intel/AMD CPUs with AVX2 and FMA support
 * - AVX: Intel/AMD CPUs with AVX support
 * - NEON: ARM CPUs with NEON support
 * - Scalar: Fallback for all other architectures
 * - OpenMP: Multi-threading support
 *
 * Build instructions:
 *     pip install pybind11
 *     python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

// OpenMP support
#ifdef USE_OPENMP
    #include <omp.h>
    #define HAS_OPENMP 1
#else
    #define HAS_OPENMP 0
#endif

// SIMD instruction set detection and headers
#if defined(USE_AVX512) || defined(__AVX512F__)
    #include <immintrin.h>
    #define SIMD_LEVEL "AVX-512"
    #define USE_SIMD 1
    #define USE_AVX512_OPT 1
#elif defined(USE_AVX2) || defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_LEVEL "AVX2"
    #define USE_SIMD 1
    #define USE_AVX2_OPT 1
#elif defined(USE_AVX) || defined(__AVX__)
    #include <immintrin.h>
    #define SIMD_LEVEL "AVX"
    #define USE_SIMD 1
    #define USE_AVX_OPT 1
#elif defined(USE_NEON) || defined(__ARM_NEON)
    #include <arm_neon.h>
    #define SIMD_LEVEL "NEON"
    #define USE_SIMD 1
    #define USE_NEON_OPT 1
#else
    #define SIMD_LEVEL "Scalar"
    #define USE_SIMD 0
#endif

// FMA support
#if defined(USE_FMA) || defined(__FMA__)
    #define HAS_FMA 1
#else
    #define HAS_FMA 0
#endif

namespace py = pybind11;

// Forward declare BLAS function if available
#ifdef USE_BLAS
extern "C" {
    void cblas_sgemm(const enum CBLAS_ORDER Order,
                    const enum CBLAS_TRANSPOSE TransA,
                    const enum CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K,
                    const float alpha, const float *A, const int lda,
                    const float *B, const int ldb,
                    const float beta, float *C, const int ldc);
}
#endif


// ============================================================================
// RMS Normalization with multi-level optimizations
// ============================================================================

py::array_t<float> rms_norm_cpp(py::array_t<float> x, py::array_t<float> weight, float eps = 1e-6f) {
    auto x_buf = x.request();
    auto w_buf = weight.request();
    
    if (x_buf.ndim < 1 || w_buf.ndim != 1) {
        throw std::runtime_error("Invalid dimensions");
    }
    
    // Get dimensions
    size_t total_size = x_buf.size;
    size_t dim = w_buf.size;
    size_t n_rows = total_size / dim;
    
    // Allocate output
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* w_ptr = static_cast<float*>(w_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    
    // Process each row (with OpenMP parallelization if available)
#ifdef USE_OPENMP
    #pragma omp parallel for if(n_rows > 4)
#endif
    for (size_t row = 0; row < n_rows; row++) {
        float* x_row = x_ptr + row * dim;
        float* out_row = out_ptr + row * dim;
        
        // Compute mean of squares with SIMD optimization
        float sum_sq = 0.0f;
        
#if defined(USE_AVX512_OPT)
        {
            // AVX-512 version (16 floats at a time)
            __m512 sum_vec = _mm512_setzero_ps();
            size_t i = 0;
            for (; i + 16 <= dim; i += 16) {
                __m512 x_vec = _mm512_loadu_ps(x_row + i);
#if HAS_FMA
                sum_vec = _mm512_fmadd_ps(x_vec, x_vec, sum_vec);
#else
                sum_vec = _mm512_add_ps(sum_vec, _mm512_mul_ps(x_vec, x_vec));
#endif
            }
            sum_sq = _mm512_reduce_add_ps(sum_vec);
            // Handle remainder
            for (; i < dim; i++) {
                sum_sq += x_row[i] * x_row[i];
            }
        }
        
#elif defined(USE_AVX2_OPT)
        {
            // AVX2 version (8 floats at a time)
            __m256 sum_vec = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 8 <= dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_row + i);
#if HAS_FMA
                sum_vec = _mm256_fmadd_ps(x_vec, x_vec, sum_vec);
#else
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x_vec, x_vec));
#endif
            }
            // Horizontal sum
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum128 = _mm_add_ps(sum_low, sum_high);
            __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            __m128 sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));
            sum_sq = _mm_cvtss_f32(sum32);
            // Handle remainder
            for (; i < dim; i++) {
                sum_sq += x_row[i] * x_row[i];
            }
        }
        
#elif defined(USE_AVX_OPT)
        {
            // AVX version (8 floats at a time, no FMA)
            __m256 sum_vec = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 8 <= dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_row + i);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x_vec, x_vec));
            }
            // Horizontal sum
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum128 = _mm_add_ps(sum_low, sum_high);
            __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            __m128 sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));
            sum_sq = _mm_cvtss_f32(sum32);
            // Handle remainder
            for (; i < dim; i++) {
                sum_sq += x_row[i] * x_row[i];
            }
        }
        
#elif defined(USE_NEON_OPT)
        {
            // ARM NEON version (4 floats at a time)
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            size_t i = 0;
            for (; i + 4 <= dim; i += 4) {
                float32x4_t x_vec = vld1q_f32(x_row + i);
                sum_vec = vmlaq_f32(sum_vec, x_vec, x_vec);  // FMA on NEON
            }
            // Horizontal sum
            float32x2_t sum_low = vget_low_f32(sum_vec);
            float32x2_t sum_high = vget_high_f32(sum_vec);
            float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
            sum_sq = vget_lane_f32(sum_pair, 0) + vget_lane_f32(sum_pair, 1);
            // Handle remainder
            for (; i < dim; i++) {
                sum_sq += x_row[i] * x_row[i];
            }
        }
        
#else
        // Scalar version
        for (size_t i = 0; i < dim; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        
        float rms = std::sqrt(sum_sq / dim + eps);
        float scale = 1.0f / rms;
        
        // Normalize and scale by weights with SIMD
#if defined(USE_AVX512_OPT)
        {
            size_t i = 0;
            __m512 scale_vec = _mm512_set1_ps(scale);
            for (; i + 16 <= dim; i += 16) {
                __m512 x_vec = _mm512_loadu_ps(x_row + i);
                __m512 w_vec = _mm512_loadu_ps(w_ptr + i);
                __m512 out_vec = _mm512_mul_ps(_mm512_mul_ps(x_vec, scale_vec), w_vec);
                _mm512_storeu_ps(out_row + i, out_vec);
            }
            for (; i < dim; i++) {
                out_row[i] = x_row[i] * scale * w_ptr[i];
            }
        }
        
#elif defined(USE_AVX2_OPT) || defined(USE_AVX_OPT)
        {
            size_t i = 0;
            __m256 scale_vec = _mm256_set1_ps(scale);
            for (; i + 8 <= dim; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x_row + i);
                __m256 w_vec = _mm256_loadu_ps(w_ptr + i);
                __m256 out_vec = _mm256_mul_ps(_mm256_mul_ps(x_vec, scale_vec), w_vec);
                _mm256_storeu_ps(out_row + i, out_vec);
            }
            for (; i < dim; i++) {
                out_row[i] = x_row[i] * scale * w_ptr[i];
            }
        }
        
#elif defined(USE_NEON_OPT)
        {
            size_t i = 0;
            float32x4_t scale_vec = vdupq_n_f32(scale);
            for (; i + 4 <= dim; i += 4) {
                float32x4_t x_vec = vld1q_f32(x_row + i);
                float32x4_t w_vec = vld1q_f32(w_ptr + i);
                float32x4_t out_vec = vmulq_f32(vmulq_f32(x_vec, scale_vec), w_vec);
                vst1q_f32(out_row + i, out_vec);
            }
            for (; i < dim; i++) {
                out_row[i] = x_row[i] * scale * w_ptr[i];
            }
        }
        
#else
        for (size_t i = 0; i < dim; i++) {
            out_row[i] = x_row[i] * scale * w_ptr[i];
        }
#endif
    }
    
    return result;
}

// ============================================================================
// SiLU activation with multi-level optimizations
// ============================================================================

py::array_t<float> silu_cpp(py::array_t<float> x) {
    auto x_buf = x.request();
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    size_t size = x_buf.size;
    
    // Process with OpenMP if available
#ifdef USE_OPENMP
    #pragma omp parallel for if(size > 8192)
#endif
    for (size_t i = 0; i < size; i++) {
        out_ptr[i] = x_ptr[i] / (1.0f + std::exp(-x_ptr[i]));
    }
    
    return result;
}


// ============================================================================
// GELU activation with multi-level optimizations
// ============================================================================

py::array_t<float> gelu_cpp(py::array_t<float> x) {
    auto x_buf = x.request();
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    size_t size = x_buf.size;
    
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    
    // Process with OpenMP if available
#ifdef USE_OPENMP
    #pragma omp parallel for if(size > 8192)
#endif
    for (size_t i = 0; i < size; i++) {
        float xi = x_ptr[i];
        float inner = sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi);
        out_ptr[i] = 0.5f * xi * (1.0f + std::tanh(inner));
    }
    
    return result;
}


// ============================================================================
// Matrix multiplication with multi-level optimizations
// ============================================================================

py::array_t<float> matmul_cpp(py::array_t<float> a, py::array_t<float> b) {
    auto a_buf = a.request();
    auto b_buf = b.request();
    
    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::runtime_error("Both arrays must be 2D");
    }
    
    size_t m = a_buf.shape[0];
    size_t k = a_buf.shape[1];
    size_t n = b_buf.shape[1];
    
    if (k != static_cast<size_t>(b_buf.shape[0])) {
        throw std::runtime_error("Incompatible dimensions");
    }
    
    auto result = py::array_t<float>({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(n)});
    auto result_buf = result.request();
    
    float* a_ptr = static_cast<float*>(a_buf.ptr);
    float* b_ptr = static_cast<float*>(b_buf.ptr);
    float* c_ptr = static_cast<float*>(result_buf.ptr);
    
    // Initialize output to zero
    std::fill(c_ptr, c_ptr + m * n, 0.0f);
    
    // Improved matmul with blocking and OpenMP
    const size_t block_size = 64;
    
#ifdef USE_OPENMP
    #pragma omp parallel for if(m > 8)
#endif
    for (size_t i = 0; i < m; i++) {
        for (size_t jb = 0; jb < n; jb += block_size) {
            size_t j_end = std::min(jb + block_size, n);
            for (size_t kb = 0; kb < k; kb += block_size) {
                size_t k_end = std::min(kb + block_size, k);
                
                // Block multiplication
                for (size_t j = jb; j < j_end; j++) {
                    float sum = c_ptr[i * n + j];
                    
#if defined(USE_AVX2_OPT) || defined(USE_AVX_OPT)
                    // SIMD inner loop for AVX/AVX2
                    __m256 sum_vec = _mm256_setzero_ps();
                    size_t p = kb;
                    for (; p + 8 <= k_end; p += 8) {
                        __m256 a_vec = _mm256_loadu_ps(a_ptr + i * k + p);
                        __m256 b_vec = _mm256_set_ps(
                            b_ptr[(p+7) * n + j], b_ptr[(p+6) * n + j],
                            b_ptr[(p+5) * n + j], b_ptr[(p+4) * n + j],
                            b_ptr[(p+3) * n + j], b_ptr[(p+2) * n + j],
                            b_ptr[(p+1) * n + j], b_ptr[p * n + j]
                        );
#if HAS_FMA
                        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
#else
                        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(a_vec, b_vec));
#endif
                    }
                    // Horizontal sum
                    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                    __m128 sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));
                    sum += _mm_cvtss_f32(sum32);
                    // Handle remainder
                    for (; p < k_end; p++) {
                        sum += a_ptr[i * k + p] * b_ptr[p * n + j];
                    }
#else
                    // Scalar inner loop
                    for (size_t p = kb; p < k_end; p++) {
                        sum += a_ptr[i * k + p] * b_ptr[p * n + j];
                    }
#endif
                    
                    c_ptr[i * n + j] = sum;
                }
            }
        }
    }
    
    return result;
}


// ============================================================================
// Capability query functions
// ============================================================================

std::string get_optimization_info() {
    std::string info = "Tensor Operations Optimization Info:\n";
    info += "  SIMD Level: " SIMD_LEVEL "\n";
    info += "  OpenMP: ";
#if HAS_OPENMP
    info += "Enabled";
#ifdef USE_OPENMP
    info += " (max threads: " + std::to_string(omp_get_max_threads()) + ")";
#endif
#else
    info += "Disabled";
#endif
    info += "\n  FMA: ";
    info += HAS_FMA ? "Enabled" : "Disabled";
    return info;
}


// ============================================================================
// Python module bindings
// ============================================================================

PYBIND11_MODULE(tensor_ops_cpp, m) {
    m.doc() = "Optimized C++ tensor operations for Distributed-Llama Python worker\n"
              "Supports multi-level CPU optimizations: AVX-512, AVX2, AVX, NEON, OpenMP";
    
    m.def("rms_norm", &rms_norm_cpp, 
          "RMS normalization with multi-level SIMD optimization",
          py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6f);
    
    m.def("silu", &silu_cpp, 
          "SiLU (Sigmoid Linear Unit) activation with OpenMP optimization");
    
    m.def("gelu", &gelu_cpp, 
          "GELU (Gaussian Error Linear Unit) activation with OpenMP optimization");
    
    m.def("matmul", &matmul_cpp, 
          "Matrix multiplication with blocking and SIMD optimization");
    
    m.def("get_optimization_info", &get_optimization_info,
          "Get information about enabled optimizations");
    
    // Capability attributes
    m.attr("has_simd") = USE_SIMD;
    m.attr("simd_level") = SIMD_LEVEL;
    m.attr("has_openmp") = HAS_OPENMP;
    m.attr("has_fma") = HAS_FMA;
    
#ifdef USE_AVX512_OPT
    m.attr("has_avx512") = true;
#else
    m.attr("has_avx512") = false;
#endif
    
#ifdef USE_AVX2_OPT
    m.attr("has_avx2") = true;
#else
    m.attr("has_avx2") = false;
#endif
    
#ifdef USE_AVX_OPT
    m.attr("has_avx") = true;
#else
    m.attr("has_avx") = false;
#endif
    
#ifdef USE_NEON_OPT
    m.attr("has_neon") = true;
#else
    m.attr("has_neon") = false;
#endif
}
