#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float pos_exp = exp(input[idx]);
    float neg_exp = exp(-input[idx]);
    if (idx < N) {
        output[idx] = (pos_exp - neg_exp) / (pos_exp + neg_exp);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}