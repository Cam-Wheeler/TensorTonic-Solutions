#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx < N) {
        float x = input[t_idx];
        output[t_idx] = x / (1 + exp(-x));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
