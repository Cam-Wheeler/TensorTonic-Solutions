#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    const int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_idx =  blockIdx.x * blockDim.x + threadIdx.x;

    if (r_idx < M && c_idx < N) {
        B[c_idx * M + r_idx] = A[r_idx * N + c_idx];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
