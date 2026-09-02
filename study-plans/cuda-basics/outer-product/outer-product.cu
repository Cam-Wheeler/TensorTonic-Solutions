#include <cuda_runtime.h>

__global__ void outer_product_kernel(const float* a, const float* b, float* C, int M, int N) {
    const int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (r_idx < M && c_idx < N) {
        C[r_idx * N + c_idx] = a[r_idx] * b[c_idx];
    }
}

extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    outer_product_kernel<<<blocks, threads>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}
