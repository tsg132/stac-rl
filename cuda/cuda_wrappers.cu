#include <cuda_runtime.h>
#include <cstddef>

// CUDA wrapper functions for C++ code
extern "C" {

void* cuda_malloc(size_t size) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void cuda_memset(void* ptr, int value, size_t size) {
    cudaMemset(ptr, value, size);
}

void cuda_set_device(int device) {
    cudaSetDevice(device);
}

} // extern "C"

namespace stac::cuda {

void zero_gradients(float* grads, int size) {
    cudaMemset(grads, 0, size * sizeof(float));
}

} // namespace stac::cuda
