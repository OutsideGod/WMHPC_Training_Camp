#include <string>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void kernel(const float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] += 2.0f * a[idx];
    }
}

int main(int argc, char* argv[]) {
    int n = std::atoi(argv[1]);
    if (n == 0) {
        printf("SUM=%.0f\n", 0.0);
    }
    else {
        double s = 0;
        cudaEvent_t start_, stop_;
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
        size_t size = (size_t)n * sizeof(float);
        float* a;
        float* b;
        CHECK_CUDA(cudaMallocManaged(&a, size));
        CHECK_CUDA(cudaMallocManaged(&b, size));
        for (int i = 0; i < n; i++) {
            a[i] = ((i % 2048)- 1024) * 0.5f;
            b[i] = (i % 1024)- 512;
        }
        int thread_per_block = 256;
        int blocks_per_grid = (n + thread_per_block - 1) / thread_per_block;
        CHECK_CUDA(cudaEventRecord(start_));
        kernel<<<blocks_per_grid, thread_per_block>>>(a, b, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop_));
        cudaDeviceSynchronize();
        for (int i = 0; i < n; i++) {
            s += b[i];
        }
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
        printf("SUM=%.0f (n=%d, time=%.3f ms)\n", s, n, ms);
        cudaFree(a);
        cudaFree(b);
        CHECK_CUDA(cudaEventDestroy(start_));
        CHECK_CUDA(cudaEventDestroy(stop_));
    }
    return 0;
}