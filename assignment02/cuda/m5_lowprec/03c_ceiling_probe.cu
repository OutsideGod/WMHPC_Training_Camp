// 问题 5.3(c):ceiling 探针。
//
// 目标:测出"5.3(b) 这种访存形状的上限带宽"。方法是写一个和你的
// quant kernel 访存完全同形(读同样的 bf16、写同样位置的 8 byte 数据
// 与 1 byte SF)、但不做任何数学的 kernel——读进来的位 xor 一下直接
// 写出去即可。它的耗时就是这个访存模式在这块卡上的地板。
//
// 跑完把三个数放在一起:探针 GB/s、你的 quant kernel GB/s(03b 的
// 输出)、两者比值。报告里回答:你的 kernel 离自己的上限还有多远,
// 差距是访存还是计算(ncu 的 SM% / DRAM% 可以佐证)。
//
// 在下面实现探针 kernel 和 launch;main 不需要修改。
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void probe_kernel(const __nv_bfloat16* __restrict__ in,
                             uint8_t* __restrict__ dataOut,
                             uint8_t* __restrict__ sfOut, int M, int K) {
    // TODO: 与你的 quant kernel 同形的访存,xor 直通,无数学。
    int groups_per_row = K / NVFP4_GROUP;
    int total_groups = M * groups_per_row;
    int num_ktiles = nvfp4_num_ktiles(K);
    for (int linear = blockIdx.x * BLOCK + threadIdx.x; linear < total_groups;
         linear += gridDim.x * BLOCK) {
        int row = linear / groups_per_row;
        int group = linear - row * groups_per_row;
        const uint4* src = reinterpret_cast<const uint4*>(
            in + (size_t)row * K + group * NVFP4_GROUP);
        uint4 a = src[0];
        uint4 b = src[1];
        uint2 out = make_uint2(a.x ^ a.z ^ b.x ^ b.z,
                               a.y ^ a.w ^ b.y ^ b.w);
        *reinterpret_cast<uint2*>(dataOut + (size_t)row * K / 2 + group * 8) =
            out;
        sfOut[sf_swizzled_offset(row, group, num_ktiles)] =
            (uint8_t)(out.x ^ out.y);
    }
}

static void launch_probe(const __nv_bfloat16* in, uint8_t* dataOut,
                         uint8_t* sfOut, int M, int K, int sms) {
    // TODO: 启动配置。
    constexpr int BLOCK = 256;
    int groups = M * (K / NVFP4_GROUP);
    int grid = (groups + BLOCK - 1) / BLOCK;
    if (grid > sms * 8) grid = sms * 8;
    probe_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);
}

int main(int argc, char** argv) {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int only = argc > 1 ? atoi(argv[1]) : -1;
    const std::vector<std::pair<int, int>> shapes = {
        {1, 4096}, {16, 4096}, {256, 4096}, {1024, 4096}, {4096, 4096},
        {16384, 4096}, {4096, 7168}, {16384, 7168}, {4096, 8192},
        {16384, 8192}};
    for (int shape_i = 0; shape_i < (int)shapes.size(); ++shape_i) {
        if (only >= 0 && shape_i != only) continue;
        const auto& shape = shapes[shape_i];
        int M = shape.first;
        int K = shape.second;
        size_t n = (size_t)M * K;
        __nv_bfloat16* dx;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, nvfp4_sf_bytes(M, K)));
        CUDA_CHECK(cudaMemset(dx, 0x3c, n * 2));
        float ms = time_avg_ms(
            [&] { launch_probe(dx, dd, dsf, M, K, sms); }, 50);
        CUDA_CHECK_KERNEL();
        double bytes = n * 2.0 + n * 0.5 + n / 16.0;
        printf("M=%-6d K=%-5d  probe %8.2f us  %6.0f GB/s\n", M, K, ms * 1e3,
               effective_gbps(bytes, ms));
        cudaFree(dx); cudaFree(dd); cudaFree(dsf);
    }
    return 0;
}
