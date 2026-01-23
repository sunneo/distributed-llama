"""
C++ Extension Module for Distributed-Llama Python Worker

This module provides Python bindings to optimized C++ implementations
of critical tensor operations:
- Matrix multiplication (using BLAS)
- RMS normalization (with SIMD)
- Activation functions (SiLU, GELU)

Build instructions:
    pip install pybind11
    c++ -O3 -Wall -shared -std=c++11 -fPIC \
        $(python3 -m pybind11 --includes) \
        -o tensor_ops_cpp$(python3-config --extension-suffix) \
        tensor_ops_cpp.cpp
"""

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <vector>

#if defined(__AVX2__)
    #include <immintrin.h>
    #define USE_SIMD 1
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define USE_SIMD 1
#else
    #define USE_SIMD 0
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


// RMS Normalization
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
    
    // Process each row
    for (size_t row = 0; row < n_rows; row++) {
        float* x_row = x_ptr + row * dim;
        float* out_row = out_ptr + row * dim;
        
        // Compute mean of squares
        float sum_sq = 0.0f;
        
#if defined(__AVX2__) && USE_SIMD
        // SIMD version for AVX2
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x_row + i);
            sum_vec = _mm256_fmadd_ps(x_vec, x_vec, sum_vec);
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
#else
        // Scalar version
        for (size_t i = 0; i < dim; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        
        float rms = std::sqrt(sum_sq / dim + eps);
        float scale = 1.0f / rms;
        
        // Normalize and scale by weights
#if defined(__AVX2__) && USE_SIMD
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
#else
        for (size_t i = 0; i < dim; i++) {
            out_row[i] = x_row[i] * scale * w_ptr[i];
        }
#endif
    }
    
    return result;
}


// SiLU activation
py::array_t<float> silu_cpp(py::array_t<float> x) {
    auto x_buf = x.request();
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    size_t size = x_buf.size;
    
#if defined(__AVX2__) && USE_SIMD
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x_ptr + i);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x_vec);
        
        // Approximate exp with polynomial
        __m256 exp_neg_x = _mm256_max_ps(neg_x, _mm256_set1_ps(-88.0f));
        exp_neg_x = _mm256_min_ps(exp_neg_x, _mm256_set1_ps(88.0f));
        
        // Use built-in approximation
        // For simplicity, fall back to scalar for now
        for (size_t j = i; j < i + 8; j++) {
            out_ptr[j] = x_ptr[j] / (1.0f + std::exp(-x_ptr[j]));
        }
    }
    for (; i < size; i++) {
        out_ptr[i] = x_ptr[i] / (1.0f + std::exp(-x_ptr[i]));
    }
#else
    for (size_t i = 0; i < size; i++) {
        out_ptr[i] = x_ptr[i] / (1.0f + std::exp(-x_ptr[i]));
    }
#endif
    
    return result;
}


// GELU activation (approximation)
py::array_t<float> gelu_cpp(py::array_t<float> x) {
    auto x_buf = x.request();
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    size_t size = x_buf.size;
    
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    
    for (size_t i = 0; i < size; i++) {
        float xi = x_ptr[i];
        float inner = sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi);
        out_ptr[i] = 0.5f * xi * (1.0f + std::tanh(inner));
    }
    
    return result;
}


// Simple matrix multiplication (for when BLAS is not available)
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
    
    // Simple matmul (row-major)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a_ptr[i * k + p] * b_ptr[p * n + j];
            }
            c_ptr[i * n + j] = sum;
        }
    }
    
    return result;
}


PYBIND11_MODULE(tensor_ops_cpp, m) {
    m.doc() = "Optimized C++ tensor operations for Distributed-Llama Python worker";
    
    m.def("rms_norm", &rms_norm_cpp, 
          "RMS normalization with SIMD optimization",
          py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6f);
    
    m.def("silu", &silu_cpp, 
          "SiLU (Sigmoid Linear Unit) activation");
    
    m.def("gelu", &gelu_cpp, 
          "GELU (Gaussian Error Linear Unit) activation");
    
    m.def("matmul", &matmul_cpp, 
          "Matrix multiplication (simple implementation, use NumPy/BLAS for production)");
    
    m.attr("has_simd") = USE_SIMD;
}
