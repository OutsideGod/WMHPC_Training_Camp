// 问题 5.3(b):实现 NVFP4 quant kernel。
//
// 输入 bf16 矩阵 [M, K](K 是 16 的倍数),输出:
//   dataOut:e2m1 打包数据,每行 K/2 byte,低 nibble 放偶数下标元素
//   sfOut:  e4m3 SF,swizzled 布局(偏移用 nvfp4_common.h 的
//           sf_swizzled_offset;整个 SF 张量已在调用前清零)
//
// 每组的计算顺序(判测按同一顺序生成真值,逐 byte 严格相等):
//   amax = 组内 16 个值的绝对值最大
//   sf8  = __nv_fp8_e4m3(amax / 6.0f)
//   sf   = float(sf8)
//   inv  = sf != 0 ? 1.0f / sf : 0.0f
//   nibble[i] = encode(v[i] * inv)
//
// 设备侧的 e2m1 转换直接用 cuda_fp4.h 的 __nv_fp4x2_e2m1(float2 的 .x
// 进低 nibble),它在 sm_100 家族上是单条硬件指令;你在 5.3(a) 写的
// 编码器语义与它一致,host 参考用的就是它。
//
// 组织建议:16 元素 = 32 byte,一个线程恰好负责一个组,天然免掉组内
// 线程协作;quant 没有行间依赖,grid 怎么铺完全自由。
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void nvfp4_quant_kernel(const __nv_bfloat16* __restrict__ in,
                                   uint8_t* __restrict__ dataOut,
                                   uint8_t* __restrict__ sfOut, int M, int K) {
    // TODO: 实现。
    int groups_per_row = K / NVFP4_GROUP;
    int total_groups = M * groups_per_row;
    int num_ktiles = nvfp4_num_ktiles(K);
    for (int linear = blockIdx.x * BLOCK + threadIdx.x; linear < total_groups;
         linear += gridDim.x * BLOCK) {
        int row = linear / groups_per_row;
        int group = linear - row * groups_per_row;
        const __nv_bfloat16* src = in + (size_t)row * K + group * NVFP4_GROUP;

        float value[NVFP4_GROUP];
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < NVFP4_GROUP; ++i) {
            value[i] = __bfloat162float(src[i]);
            amax = fmaxf(amax, fabsf(value[i]));
        }
        __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(amax / 6.0f);
        float sf = float(sf8);
        float inv = sf != 0.f ? 1.0f / sf : 0.f;
        sfOut[sf_swizzled_offset(row, group, num_ktiles)] =
            *reinterpret_cast<uint8_t*>(&sf8);

        uint8_t packed[8];
#pragma unroll
        for (int i = 0; i < NVFP4_GROUP; i += 2) {
            __nv_fp4x2_e2m1 pair(make_float2(value[i] * inv,
                                              value[i + 1] * inv));
            packed[i / 2] = *reinterpret_cast<uint8_t*>(&pair);
        }
        *reinterpret_cast<uint2*>(dataOut + (size_t)row * K / 2 + group * 8) =
            *reinterpret_cast<uint2*>(packed);
    }
}

// 判测和 5.4 会按这个签名调用;grid 大小你自己定,写在这里。
inline void launch_nvfp4_quant(const __nv_bfloat16* in, uint8_t* dataOut,
                               uint8_t* sfOut, int M, int K, int sms) {
    // TODO: 选择 grid/block 并启动 nvfp4_quant_kernel。
    constexpr int BLOCK = 256;
    int groups = M * (K / NVFP4_GROUP);
    int grid = (groups + BLOCK - 1) / BLOCK;
    if (grid > sms * 8) grid = sms * 8;
    nvfp4_quant_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);
}
