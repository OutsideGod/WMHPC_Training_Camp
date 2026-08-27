#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = 32;



__device__ void load_A(const __nv_fp8_e4m3* A, uint32_t a[4]) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tig = lane & 3;
    int row = 0;
    int col = 0;
    for (int i = 0; i < 4; i++) {
        row = (i & 1) * 8 + gid;
        col = (i >> 1) * 16 + tig * 4;
        a[i] = *reinterpret_cast<const uint32_t*>(&A[row * K + col]);
    }
}

__device__ void load_B(const __nv_fp8_e4m3* B, uint32_t b[2]) {
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tig = lane & 3;
    int k = 0;
    int n = 0;
    for (int i = 0; i < 2; i++) {
        k = i * 16 + tig * 4;
        n = gid;
        b[i] = *reinterpret_cast<const uint32_t*>(&B[n * K + k]);
    }
}

__global__ void mma_fp8_kernel(const __nv_fp8_e4m3* dA, const __nv_fp8_e4m3* dB, float* dD) {
    __shared__ __nv_fp8_e4m3 sA[M * K];
    __shared__ __nv_fp8_e4m3 sB[N * K];
    uint32_t a[4];
    uint32_t b[2];
    int tid = threadIdx.x;

    for (int i = tid; i < M * K; i += blockDim.x) {
        sA[i] = dA[i];
    }
    for (int i = tid; i < K * N; i += blockDim.x) {
        sB[i] = dB[i];
    }
    __syncthreads();
    load_A(sA, a);
    load_B(sB, b);

    float c[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4];

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
    int lane = threadIdx.x & 31;
    int gid = lane >> 2;
    int tig = lane & 3;
    for (int i = 0; i < 4; i++) {
        dD[(i >> 1) * 64 + gid * 8 + tig * 2 + (i & 1)] = d[i]; 
    }
}

int main(int argc, char** argv) {

    int seed = std::atoi(argv[1]);
    std::srand(seed);

    __nv_fp8_e4m3 hA[M * K];
    __nv_fp8_e4m3 hB[K * N];

    float hD[M * N] = {};
    float ref[M * N] = {};

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float x = float(std::rand() % 5 - 2);
            hA[m * K + k] = __nv_fp8_e4m3(x);
        }
    }

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float x = float(std::rand() % 5 - 2);
            hB[n * K + k] = __nv_fp8_e4m3(x);
        }
    }

    __nv_fp8_e4m3* dA = nullptr;
    __nv_fp8_e4m3* dB = nullptr;
    float* dD = nullptr;

    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dD, sizeof(hD));

    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemset(dD, 0, sizeof(hD));

    mma_fp8_kernel<<<1, 32>>>(dA, dB, dD);

    cudaDeviceSynchronize();

    cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost);

    // CPU reference
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = float(hA[m * K + k]);
                float b = float(hB[n * K + k]);
                acc += a * b;
            }
            ref[m * N + n] = acc;
        }
    }

    for (int i = 0; i < M * N; ++i) {
        if (hD[i] != ref[i]) {
            printf("MISMATCH idx=%d gpu=%f ref=%f\n",
                   i, hD[i], ref[i]);

            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dD);
            return 1;
        }
    }

    printf("PASS seed=%d\n", seed);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);

    return 0;
}