/*
 * OpenCL Backend for Tensor Operations
 * 
 * GPU-accelerated tensor operations using OpenCL.
 * This is an optional backend that can be built when OpenCL is available.
 * Works on NVIDIA, AMD, Intel GPUs, and even CPUs.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// OpenCL kernel source for RMS norm
const char* rms_norm_kernel_src = R"(
__kernel void rms_norm(__global const float* x,
                       __global const float* weight,
                       __global float* out,
                       int dim,
                       float eps) {
    int row = get_global_id(0);
    
    __global const float* x_row = x + row * dim;
    __global float* out_row = out + row * dim;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x_row[i] * x_row[i];
    }
    
    float rms = sqrt(sum_sq / dim + eps);
    float scale = 1.0f / rms;
    
    // Normalize and apply weights
    for (int i = 0; i < dim; i++) {
        out_row[i] = x_row[i] * scale * weight[i];
    }
}
)";

// OpenCL kernel source for SiLU
const char* silu_kernel_src = R"(
__kernel void silu(__global const float* x,
                   __global float* out) {
    int i = get_global_id(0);
    float val = x[i];
    out[i] = val / (1.0f + exp(-val));
}
)";

// OpenCL kernel source for GELU
const char* gelu_kernel_src = R"(
__kernel void gelu(__global const float* x,
                   __global float* out) {
    int i = get_global_id(0);
    const float sqrt_2_over_pi = 0.7978845608f;
    float xi = x[i];
    float inner = sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi);
    out[i] = 0.5f * xi * (1.0f + tanh(inner));
}
)";

// Simple OpenCL context manager
class OpenCLContext {
public:
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    
    OpenCLContext() {
        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, NULL);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        queue = clCreateCommandQueue(context, device, 0, NULL);
    }
    
    ~OpenCLContext() {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    
    cl_program build_program(const char* source) {
        cl_int err;
        cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create program");
        
        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            // Get build log
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
            throw std::runtime_error("Build failed: " + std::string(log.data()));
        }
        
        return program;
    }
};

static OpenCLContext* g_context = nullptr;

void init_opencl() {
    if (!g_context) {
        g_context = new OpenCLContext();
    }
}

py::array_t<float> rms_norm_opencl(py::array_t<float> x, py::array_t<float> weight, float eps = 1e-6f) {
    init_opencl();
    
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
    
    // Build program
    cl_program program = g_context->build_program(rms_norm_kernel_src);
    cl_kernel kernel = clCreateKernel(program, "rms_norm", NULL);
    
    // Create buffers
    cl_mem d_x = clCreateBuffer(g_context->context, CL_MEM_READ_ONLY, total_size * sizeof(float), NULL, NULL);
    cl_mem d_w = clCreateBuffer(g_context->context, CL_MEM_READ_ONLY, dim * sizeof(float), NULL, NULL);
    cl_mem d_out = clCreateBuffer(g_context->context, CL_MEM_WRITE_ONLY, total_size * sizeof(float), NULL, NULL);
    
    // Copy data to device
    clEnqueueWriteBuffer(g_context->queue, d_x, CL_TRUE, 0, total_size * sizeof(float), x_ptr, 0, NULL, NULL);
    clEnqueueWriteBuffer(g_context->queue, d_w, CL_TRUE, 0, dim * sizeof(float), w_ptr, 0, NULL, NULL);
    
    // Set kernel arguments
    int dim_int = (int)dim;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_w);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 3, sizeof(int), &dim_int);
    clSetKernelArg(kernel, 4, sizeof(float), &eps);
    
    // Execute kernel
    size_t global_size = n_rows;
    clEnqueueNDRangeKernel(g_context->queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    
    // Read back results
    clEnqueueReadBuffer(g_context->queue, d_out, CL_TRUE, 0, total_size * sizeof(float), out_ptr, 0, NULL, NULL);
    
    // Cleanup
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_w);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    
    return result;
}

py::array_t<float> silu_opencl(py::array_t<float> x) {
    init_opencl();
    
    auto x_buf = x.request();
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    size_t size = x_buf.size;
    
    cl_program program = g_context->build_program(silu_kernel_src);
    cl_kernel kernel = clCreateKernel(program, "silu", NULL);
    
    cl_mem d_x = clCreateBuffer(g_context->context, CL_MEM_READ_ONLY, size * sizeof(float), NULL, NULL);
    cl_mem d_out = clCreateBuffer(g_context->context, CL_MEM_WRITE_ONLY, size * sizeof(float), NULL, NULL);
    
    clEnqueueWriteBuffer(g_context->queue, d_x, CL_TRUE, 0, size * sizeof(float), x_ptr, 0, NULL, NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    
    size_t global_size = size;
    clEnqueueNDRangeKernel(g_context->queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(g_context->queue, d_out, CL_TRUE, 0, size * sizeof(float), out_ptr, 0, NULL, NULL);
    
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    
    return result;
}

py::array_t<float> gelu_opencl(py::array_t<float> x) {
    init_opencl();
    
    auto x_buf = x.request();
    auto result = py::array_t<float>(x_buf.shape);
    auto result_buf = result.request();
    
    float* x_ptr = static_cast<float*>(x_buf.ptr);
    float* out_ptr = static_cast<float*>(result_buf.ptr);
    size_t size = x_buf.size;
    
    cl_program program = g_context->build_program(gelu_kernel_src);
    cl_kernel kernel = clCreateKernel(program, "gelu", NULL);
    
    cl_mem d_x = clCreateBuffer(g_context->context, CL_MEM_READ_ONLY, size * sizeof(float), NULL, NULL);
    cl_mem d_out = clCreateBuffer(g_context->context, CL_MEM_WRITE_ONLY, size * sizeof(float), NULL, NULL);
    
    clEnqueueWriteBuffer(g_context->queue, d_x, CL_TRUE, 0, size * sizeof(float), x_ptr, 0, NULL, NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    
    size_t global_size = size;
    clEnqueueNDRangeKernel(g_context->queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(g_context->queue, d_out, CL_TRUE, 0, size * sizeof(float), out_ptr, 0, NULL, NULL);
    
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    
    return result;
}

std::string get_opencl_info() {
    try {
        init_opencl();
        
        char device_name[128];
        clGetDeviceInfo(g_context->device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
        
        std::string info = "OpenCL Backend Info:\n";
        info += "  Device: " + std::string(device_name) + "\n";
        
        return info;
    } catch (...) {
        return "OpenCL Backend Info:\n  Error: Failed to initialize OpenCL\n";
    }
}

PYBIND11_MODULE(tensor_ops_opencl, m) {
    m.doc() = "OpenCL-accelerated tensor operations";
    
    m.def("rms_norm", &rms_norm_opencl, 
          "RMS normalization on OpenCL device",
          py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6f);
    
    m.def("silu", &silu_opencl, 
          "SiLU activation on OpenCL device");
    
    m.def("gelu", &gelu_opencl, 
          "GELU activation on OpenCL device");
    
    m.def("get_opencl_info", &get_opencl_info,
          "Get OpenCL device information");
    
    m.attr("backend") = "OpenCL";
}
