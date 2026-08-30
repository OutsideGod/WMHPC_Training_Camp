#!/usr/bin/env python3
"""Single-call FlashKDA driver for Nsight Compute.

Unlike the official benchmark, this does not also launch FLA/GDN kernels and
does not loop over state variants. One invocation produces exactly one K1
prepare and one K2 recurrence launch, keeping `ncu --set full` tractable.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

import flash_kda


CASES = {
    "fixed": [8192],
    "varlen_mixed": [1300, 547, 2048, 963, 271, 3063],
    "varlen_8x1024": [1024] * 8,
}


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=CASES, default="fixed")
    p.add_argument("--heads", type=int, default=96)
    p.add_argument("--seed", type=int, default=20260830)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    seq_lens = CASES[args.case]
    total = sum(seq_lens)
    n_seq = len(seq_lens)
    h, d = args.heads, 128

    shape = (1, total, h, d)
    q = F.normalize(torch.randn(shape, dtype=torch.float32, device=device), dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape, dtype=torch.float32, device=device), dim=-1).bfloat16()
    v = torch.randn(shape, dtype=torch.bfloat16, device=device)
    g = torch.randn(shape, dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, total, h), dtype=torch.bfloat16, device=device)
    a_log = torch.rand(h, dtype=torch.float32, device=device)
    dt_bias = torch.rand((h, d), dtype=torch.float32, device=device)
    initial_state = torch.randn((n_seq, h, d, d), dtype=torch.float32, device=device)
    final_state = torch.empty_like(initial_state)
    out = torch.empty_like(q)

    kwargs = {}
    if n_seq > 1:
        prefix = [0]
        for length in seq_lens:
            prefix.append(prefix[-1] + length)
        kwargs["cu_seqlens"] = torch.tensor(prefix, dtype=torch.int64, device=device)

    flash_kda.fwd(
        q, k, v, g, beta, 1.0 / math.sqrt(d), out,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=initial_state,
        final_state=final_state,
        **kwargs,
    )
    torch.cuda.synchronize()
    print(
        f"case={args.case} T={total} H={h} N={n_seq} "
        f"out_checksum={out.float().sum().item():.6e} "
        f"state_checksum={final_state.sum().item():.6e}"
    )


if __name__ == "__main__":
    main()
