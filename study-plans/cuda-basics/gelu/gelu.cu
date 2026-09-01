#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx < N) {
        output[t_idx] = (0.5 * input[t_idx]) * (1 + erf(input[t_idx] / sqrt(2.f)));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
