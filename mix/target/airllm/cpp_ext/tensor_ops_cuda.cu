/*
 * CUDA Backend for Tensor Operations
 * 
 * GPU-accelerated tensor operations using NVIDIA CUDA.
 * This is an optional backend that can be built when CUDA is available.
 *
 * Build instructions:
 *     nvcc -O3 -shared -Xcompiler -fPIC \
 *         $(python3 -m pybind11 --includes) \
 *         -o tensor_ops_cuda$(python3-config --extension-suffix) \
 *         tensor_ops_cuda.cu
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

// CUDA kernel for RMS normalization
__global__ void rms_norm_kernel(const float* x, const float* weight, float* out, 
                                int dim, int n_rows, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    
    const float* x_row = x + row * dim;
    float* out_row = out + row * dim;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x_row[i] * x_row[i];
    }
    
    float rms = sqrtf(sum_sq / dim + eps);
    float scale = 1.0f / rms;
    
    // Normalize and apply weights
    for (int i = 0; i < dim; i++) {
        out_row[i] = x_row[i] * scale * weight[i];
    }
}

// CUDA kernel for SiLU activation
__global__ void silu_kernel(const float* x, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    
    float val = x[i];
    out[i] = val / (1.0f + expf(-val));
}

// CUDA kernel for GELU activation
__global__ void gelu_kernel(const float* x, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    
    const float sqrt_2_over_pi = 0.7978845608f;
    float xi = x[i];
    float inner = sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi);
    out[i] = 0.5f * xi * (1.0f + tanhf(inner));
}

// Python bindings
py::array_t<float> rms_norm_cuda(py::array_t<float> x, py::array_t<float> weight, float eps = 1e-6f) {
    auto x_buf = x.request();
    auto w_buf = weight.request();
    
    size_t total_size = x_buf.size;
    size_t dim = w_buf.size;
    size_t n_rows = total_size / dim;
    
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* w_ptr = static_cast<float*>(w_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    
    // Allocate device memory
    float *d_x, *d_w, *d_out;
    cudaMalloc(&d_x, total_size * sizeof(float));
    cudaMalloc(&d_w, dim * sizeof(float));
    cudaMalloc(&d_out, total_size * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_x, x_ptr, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w_ptr, dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads = 256;
    int blocks = (n_rows + threads - 1) / threads;
    rms_norm_kernel<<<blocks, threads>>>(d_x, d_w, d_out, dim, n_rows, eps);
    
    // Copy back to host
    cudaMemcpy(out_ptr, d_out, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);
    
    return result;
}

py::array_t<float> silu_cuda(py::array_t<float> x) {
    auto x_buf = x.request();
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    size_t size = x_buf.size;
    
    float *d_x, *d_out;
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    
    cudaMemcpy(d_x, x_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    silu_kernel<<<blocks, threads>>>(d_x, d_out, size);
    
    cudaMemcpy(out_ptr, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_out);
    
    return result;
}

py::array_t<float> gelu_cuda(py::array_t<float> x) {
    auto x_buf = x.request();
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    size_t size = x_buf.size;
    
    float *d_x, *d_out;
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    
    cudaMemcpy(d_x, x_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(d_x, d_out, size);
    
    cudaMemcpy(out_ptr, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_out);
    
    return result;
}

std::string get_cuda_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    std::string info = "CUDA Backend Info:\n";
    info += "  Device count: " + std::to_string(device_count) + "\n";
    
    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        info += "  Device 0: " + std::string(prop.name) + "\n";
        info += "  Compute capability: " + std::to_string(prop.major) + "." + std::to_string(prop.minor) + "\n";
        info += "  Global memory: " + std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB\n";
    }
    
    return info;
}

PYBIND11_MODULE(tensor_ops_cuda, m) {
    m.doc() = "CUDA-accelerated tensor operations";
    
    m.def("rms_norm", &rms_norm_cuda, 
          "RMS normalization on CUDA GPU",
          py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6f);
    
    m.def("silu", &silu_cuda, 
          "SiLU activation on CUDA GPU");
    
    m.def("gelu", &gelu_cuda, 
          "GELU activation on CUDA GPU");
    
    m.def("get_cuda_info", &get_cuda_info,
          "Get CUDA device information");
    
    m.attr("backend") = "CUDA";
}
