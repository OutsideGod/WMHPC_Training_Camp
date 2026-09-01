#!/usr/bin/env python3
"""C1 benchmark with an explicit, fair FLA Triton comparison.

Unlike the upstream protocol, the FLA row uses ``safe_gate=True`` (the same
bounded gate semantics as FlashKDA), runs in inference mode, and disables the
FlashKDA backend before importing FLA.  Use ``--skip-fla`` when only comparing
two compiled FlashKDA variants.
"""

from __future__ import annotations

import argparse
import math
import os

os.environ["FLA_FLASH_KDA"] = "0"

import torch
import torch.nn.functional as F

import flash_kda
from fla_kda_ref import chunk_kda


CASES = {
    "fixed": [8192],
    "mixed": [1300, 547, 2048, 963, 271, 3063],
    "8x1024": [1024] * 8,
}


def bench(fn, warmup: int, iters: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        samples.extend(start.elapsed_time(end) for start, end in zip(starts, ends))
    return sum(samples) / len(samples), min(samples), max(samples)


def prefix(seq_lens: list[int]) -> torch.Tensor | None:
    if len(seq_lens) == 1:
        return None
    values = [0]
    for length in seq_lens:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int64, device="cuda")


def report(label: str, fn, args) -> None:
    mean, minimum, maximum = bench(fn, args.warmup, args.iters, args.repeats)
    print(f"  {label:34s} mean={mean:.4f} ms min={minimum:.4f} ms max={maximum:.4f} ms")


@torch.inference_mode()
def run_case(name: str, seq_lens: list[int], args) -> None:
    torch.manual_seed(args.seed)
    total, heads, dim = sum(seq_lens), args.heads, 128
    shape = (1, total, heads, dim)
    q = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(heads, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand((heads, dim), dtype=torch.float32, device="cuda")
    n_seq = len(seq_lens)
    state = torch.randn((n_seq, heads, dim, dim), dtype=torch.float32, device="cuda")
    final = torch.empty_like(state)
    out = torch.empty_like(v)
    cu = prefix(seq_lens)
    extra = {"cu_seqlens": cu} if cu is not None else {}
    scale = 1.0 / math.sqrt(dim)

    def flash_fp32():
        flash_kda.fwd(
            q, k, v, g, beta, scale, out,
            A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
            initial_state=state, final_state=final, **extra,
        )

    def flash_no_state():
        flash_kda.fwd(
            q, k, v, g, beta, scale, out,
            A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0, **extra,
        )

    def fla_safe():
        chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=state, output_final_state=True,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True, lower_bound=-5.0,
            A_log=a_log, dt_bias=dt_bias,
            state_v_first=True, **extra,
        )

    print(
        f"case={name} seq_lens={seq_lens} H={heads} T={total} "
        f"warmup={args.warmup} samples={args.iters * args.repeats}"
    )
    report("flash_kda fp32 state", flash_fp32, args)
    report("flash_kda no state", flash_no_state, args)
    if not args.skip_fla:
        report("FLA Triton chunk_kda safe_gate", fla_safe, args)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, default=96)
    p.add_argument("--case", choices=["all", *CASES], default="all")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--skip-fla", action="store_true")
    args = p.parse_args()
    if args.heads <= 0:
        p.error("--heads must be positive")

    import flash_kda_C

    get_flag = getattr(flash_kda_C, "k2_vsplit_enabled", None)
    flag = get_flag() if get_flag is not None else "upstream-uninstrumented"
    print(f"k2_vsplit_enabled={flag}")
    print("benchmark_mode=inference FLA_FLASH_KDA=0 safe_gate=True")
    selected = CASES.items() if args.case == "all" else [(args.case, CASES[args.case])]
    for name, seq_lens in selected:
        run_case(name, seq_lens, args)


if __name__ == "__main__":
    main()
