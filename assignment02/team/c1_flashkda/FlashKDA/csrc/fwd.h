#pragma once
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

// Experimental C1 challenge path.  The default build remains bit-for-bit on
// the original one-CTA-per-(sequence, head) K2 launch.  Set the compile-time
// flag to 1 to launch two independent CTAs, each owning 64 value columns.
#ifndef FLASH_KDA_K2_VSPLIT
#define FLASH_KDA_K2_VSPLIT 0
#endif

static_assert(FLASH_KDA_K2_VSPLIT == 0 || FLASH_KDA_K2_VSPLIT == 1,
              "FLASH_KDA_K2_VSPLIT must be 0 or 1");

template <int D, bool HasStateIn = true, bool HasStateOut = true, bool StateFP32 = false, bool IsVarlen = true>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    cudaStream_t stream
);
