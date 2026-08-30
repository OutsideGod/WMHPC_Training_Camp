// C1 microbenchmark: can a tcgen05 instruction profitably replace the
// m16n16k16 building block used throughout FlashKDA without changing CHUNK=16?
//
// The SM80 path computes exactly m16n16k16.  Dense bf16 tcgen05 has M>=64, so
// the SM100 path pads M from 16 to 64 and only keeps the first 16 result rows.
// Both kernels repeat the same A@B operation `inner_iters` times so that launch
// overhead is amortized.  The program reports both physical and useful FLOP/s;
// useful FLOP/s is the fair number for FlashKDA.
//
// Build on B300 (CUDA 13):
//   nvcc -O3 -std=c++17 \
//     -gencode arch=compute_103a,code=sm_103a \
//     -o mma16_vs_tcgen05 mma16_vs_tcgen05.cu
// Run:
//   ./mma16_vs_tcgen05 [inner_iters=4096] [launches=200]

// This is deliberately an instruction-granularity experiment, not a proposed
// end-to-end kernel.  tcgen05 setup/TMEM drain are included because a direct
// substitution in FlashKDA would have to pay them.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(err_));                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

constexpr int LOGICAL_M = 16;
constexpr int N = 16;
constexpr int K = 16;
constexpr int TCGEN_M = 64;
constexpr int PAD_K = 64;  // makes the 128-B-swizzled descriptor convenient

__host__ __device__ inline int swz128(int row, int col_byte) {
    int atom = row >> 3;
    int r = row & 7;
    int chunk = col_byte >> 4;
    int in16 = col_byte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3fff);
    d |= (uint64_t)((lbo >> 4) & 0x3fff) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3fff) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
    }
}

__global__ void mma_sync_16(const __nv_bfloat16* a,
                            const __nv_bfloat16* b_col_major, float* out,
                            int inner_iters) {
    using namespace nvcuda;
    __shared__ __nv_bfloat16 sa[LOGICAL_M * K];
    __shared__ __nv_bfloat16 sb[K * N];
    for (int i = threadIdx.x; i < LOGICAL_M * K; i += blockDim.x) sa[i] = a[i];
    for (int i = threadIdx.x; i < K * N; i += blockDim.x) sb[i] = b_col_major[i];
    __syncthreads();

    if (threadIdx.x < 32) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major>
            af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::col_major>
            bf;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
        wmma::load_matrix_sync(af, sa, K);
        wmma::load_matrix_sync(bf, sb, K);
        wmma::fill_fragment(cf, 0.0f);
        for (int i = 0; i < inner_iters; ++i) wmma::mma_sync(cf, af, bf, cf);
        wmma::store_matrix_sync(out, cf, N, wmma::mem_row_major);
    }
}

__global__ void tcgen05_padded_64(const __nv_bfloat16* a,
                                  const __nv_bfloat16* b_col_major,
                                  float* out, int inner_iters) {
    // K-major matrices with 128-B swizzle.  Only A rows [0,16) and K [0,16)
    // are useful; padding is explicitly zeroed.
    __shared__ __align__(1024) uint8_t sa[TCGEN_M * PAD_K * 2];
    __shared__ __align__(1024) uint8_t sb[N * PAD_K * 2];
    __shared__ __align__(8) uint64_t mbar;
    __shared__ uint32_t taddr_s[1];

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);

    if (warp == 0 && lane == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : : "r"(mbar_addr),
                     "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    if (warp == 0) {
        uint32_t dst = (uint32_t)__cvta_generic_to_shared(taddr_s);
        // Allocation is in TMEM columns; N=16 still uses the 32-column quantum.
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;"
            :
            : "r"(dst), "r"(32));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    for (int i = tid; i < TCGEN_M * PAD_K; i += blockDim.x) {
        int m = i / PAD_K;
        int k = i % PAD_K;
        __nv_bfloat16 x = __float2bfloat16(0.0f);
        if (m < LOGICAL_M && k < K) x = a[m * K + k];
        *reinterpret_cast<__nv_bfloat16*>(&sa[swz128(m, k * 2)]) = x;
    }
    for (int i = tid; i < N * PAD_K; i += blockDim.x) {
        int n = i / PAD_K;
        int k = i % PAD_K;
        __nv_bfloat16 x = __float2bfloat16(0.0f);
        if (k < K) x = b_col_major[n * K + k];
        *reinterpret_cast<__nv_bfloat16*>(&sb[swz128(n, k * 2)]) = x;
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    uint32_t elected = 0;
    asm volatile(
        "{\n.reg .pred p;\nelect.sync _|p, 0xffffffff;\n"
        "selp.b32 %0, 1, 0, p;\n}"
        : "=r"(elected));
    uint32_t taddr = taddr_s[0];
    if (warp == 0 && elected) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        uint32_t abase = (uint32_t)__cvta_generic_to_shared(sa);
        uint32_t bbase = (uint32_t)__cvta_generic_to_shared(sb);
        uint64_t da = make_desc_sm100(abase, 0, 1024, 2);
        uint64_t db = make_desc_sm100(bbase, 0, 1024, 2);
        // f32 C, bf16 A/B, N=16, M=64, K=16.
        uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                         (2u << 17) | (4u << 24);
        for (int i = 0; i < inner_iters; ++i) {
            uint32_t accum = i != 0;
            asm volatile(
                "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                :
                : "r"(taddr), "l"(da), "l"(db), "r"(idesc), "r"(accum));
        }
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
            ".shared::cluster.b64 [%0];"
            :
            : "r"(mbar_addr)
            : "memory");
    }
    mbar_wait(mbar_addr, 0);

    asm volatile("tcgen05.fence::after_thread_sync;");
    // TMEM is addressed in 32-row lanes.  Only warp 0 is needed to drain the
    // useful rows [0, 16); asking warp 1 to read padded rows [32, 64) would
    // charge tcgen05 for work that a direct CHUNK=16 replacement can omit.
    if (warp == 0) {
        for (int c = 0; c < N; c += 8) {
            uint32_t src = taddr + ((uint32_t)(warp * 32) << 16) + c;
            float r[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]),
                  "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
                : "r"(src));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int row = warp * 32 + lane;
            if (row < LOGICAL_M) {
#pragma unroll
                for (int x = 0; x < 8; ++x) out[row * N + c + x] = r[x];
            }
        }
    }

    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :
                     : "r"(taddr), "r"(32));
    }
}

template <class Launch>
float time_ms(Launch launch, int launches) {
    for (int i = 0; i < 20; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < launches; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total / launches;
}

int main(int argc, char** argv) {
    int inner_iters = argc > 1 ? std::atoi(argv[1]) : 4096;
    int launches = argc > 2 ? std::atoi(argv[2]) : 200;
    if (inner_iters <= 0 || launches <= 0) return 2;

    std::vector<__nv_bfloat16> ha(LOGICAL_M * K), hb(K * N);
    for (int m = 0; m < LOGICAL_M; ++m)
        for (int k = 0; k < K; ++k)
            ha[m * K + k] = __float2bfloat16(float((m + 2 * k) % 5 - 2));
    // Stored col-major: hb[n*K+k].
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            hb[n * K + k] = __float2bfloat16(float((3 * n + k) % 5 - 2));

    __nv_bfloat16 *da, *db;
    float *d_mma, *d_tcgen;
    CUDA_CHECK(cudaMalloc(&da, ha.size() * sizeof(*da)));
    CUDA_CHECK(cudaMalloc(&db, hb.size() * sizeof(*db)));
    CUDA_CHECK(cudaMalloc(&d_mma, LOGICAL_M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tcgen, LOGICAL_M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), ha.size() * sizeof(*da),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), hb.size() * sizeof(*db),
                          cudaMemcpyHostToDevice));

    auto launch_mma = [&] {
        mma_sync_16<<<1, 128>>>(da, db, d_mma, inner_iters);
    };
    auto launch_tcgen = [&] {
        tcgen05_padded_64<<<1, 128>>>(da, db, d_tcgen, inner_iters);
    };
    launch_mma();
    launch_tcgen();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> mma(LOGICAL_M * N), tcgen(LOGICAL_M * N);
    CUDA_CHECK(cudaMemcpy(mma.data(), d_mma, mma.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tcgen.data(), d_tcgen, tcgen.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float max_diff = 0.0f;
    bool finite = true;
    for (size_t i = 0; i < mma.size(); ++i) {
        finite = finite && std::isfinite(mma[i]) && std::isfinite(tcgen[i]);
        max_diff = std::fmax(max_diff, std::fabs(mma[i] - tcgen[i]));
    }

    float mma_ms = time_ms(launch_mma, launches);
    float tcgen_ms = time_ms(launch_tcgen, launches);
    double useful_flop = double(inner_iters) * 2.0 * LOGICAL_M * N * K;
    double physical_tcgen_flop = double(inner_iters) * 2.0 * TCGEN_M * N * K;
    auto tflops = [](double flop, float ms) { return flop / (ms * 1.0e9); };

    std::printf("inner_iters=%d launches=%d finite=%s max_diff=%.6g\n",
                inner_iters, launches, finite ? "true" : "false", max_diff);
    std::printf("mma.sync exact m16n16k16 : %8.3f us  useful=%8.3f TFLOP/s\n",
                mma_ms * 1e3f, tflops(useful_flop, mma_ms));
    std::printf("tcgen05 padded m64n16k16: %8.3f us  physical=%8.3f TFLOP/s "
                "useful=%8.3f TFLOP/s\n",
                tcgen_ms * 1e3f, tflops(physical_tcgen_flop, tcgen_ms),
                tflops(useful_flop, tcgen_ms));
    std::printf("useful speedup tcgen05/mma.sync: %.3fx; utilization=25%%\n",
                mma_ms / tcgen_ms);

    cudaFree(da);
    cudaFree(db);
    cudaFree(d_mma);
    cudaFree(d_tcgen);
    return finite && max_diff == 0.0f ? 0 : 1;
}
