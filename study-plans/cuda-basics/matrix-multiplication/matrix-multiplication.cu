#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (r_idx < M && c_idx < N) {
        float thread_sum = 0.f;
        for (int dot_idx = 0; dot_idx < K; dot_idx++) {
            thread_sum += A[r_idx * K + dot_idx] * B[dot_idx * N + c_idx];
        }
        C[r_idx * N + c_idx] = thread_sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
