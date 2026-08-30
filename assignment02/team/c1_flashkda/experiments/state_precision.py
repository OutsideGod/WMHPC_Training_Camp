#!/usr/bin/env python3
"""Isolate the accuracy cost of storing FlashKDA's recurrent state in bf16.

This is not a second KDA implementation.  It exercises the exact state-level
shape and recurrence pattern used by K2:

    S_next = decay[:, None] * S + delta_S

The update is evaluated in fp32 and only the persistent state is rounded to
bf16, matching FlashKDA's design.  A float64 recurrence is the reference and a
float32-state recurrence is the control.  Run the upstream end-to-end tests as
well before making a model-quality claim.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

import torch


def relative_rms(actual: torch.Tensor, reference: torch.Tensor) -> float:
    diff = (actual.double() - reference).square().mean().sqrt()
    base = reference.square().mean().sqrt().clamp_min(1e-30)
    return (diff / base).item()


def run_case(
    *, d: int, chunks: int, chunk: int, gate_per_token: float, seed: int
) -> dict[str, float | int | str]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ref = torch.zeros((d, d), dtype=torch.float64)
    fp32 = torch.zeros((d, d), dtype=torch.float32)
    bf16 = torch.zeros((d, d), dtype=torch.bfloat16)
    probe_ref = torch.empty(0, dtype=torch.float64)
    probe_fp32 = torch.empty(0, dtype=torch.float32)
    probe_bf16 = torch.empty(0, dtype=torch.float32)

    for _ in range(chunks):
        # Per-feature variation avoids turning the test into a scalar decay.
        jitter = 0.25 * torch.randn(d, generator=gen, dtype=torch.float64)
        log_decay = chunk * gate_per_token * (1.0 + jitter).clamp(0.25, 1.75)
        decay64 = torch.exp(log_decay)
        decay32 = decay64.float()

        # A rank-16 update has the same algebraic form as k_restored.T @ U.
        k = torch.randn((d, chunk), generator=gen, dtype=torch.float32) / math.sqrt(d)
        u = torch.randn((chunk, d), generator=gen, dtype=torch.float32) / math.sqrt(chunk)
        delta32 = 0.05 * (k @ u)
        delta64 = delta32.double()

        ref = decay64[:, None] * ref + delta64
        fp32 = decay32[:, None] * fp32 + delta32
        # Multiplication/addition are fp32; bf16 is only the between-chunk store.
        bf16 = (decay32[:, None] * bf16.float() + delta32).to(torch.bfloat16)

        q = torch.randn((8, d), generator=gen, dtype=torch.float32) / math.sqrt(d)
        probe_ref = q.double() @ ref
        probe_fp32 = q @ fp32
        probe_bf16 = q @ bf16.float()

    return {
        "tokens": chunks * chunk,
        "chunks": chunks,
        "gate/token": gate_per_token,
        "final_fp32_rel_rms": relative_rms(fp32, ref),
        "final_bf16_rel_rms": relative_rms(bf16.float(), ref),
        "probe_fp32_rel_rms": relative_rms(probe_fp32, probe_ref),
        "probe_bf16_rel_rms": relative_rms(probe_bf16, probe_ref),
        "bf16_max_abs": (bf16.double() - ref).abs().max().item(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260829)
    p.add_argument("--output", help="optional CSV path; stdout when omitted")
    args = p.parse_args()
    torch.set_num_threads(1)

    # weak decay is the accumulation-heavy stress case; stronger decay limits
    # the time horizon over which bf16 rounding can accumulate.
    cases = []
    for chunks in (64, 128, 512):
        for gate in (-1e-4, -2e-2, -5e-1):
            cases.append(
                run_case(
                    d=args.d,
                    chunks=chunks,
                    chunk=args.chunk,
                    gate_per_token=gate,
                    seed=args.seed + chunks + round(abs(gate) * 10000),
                )
            )

    stream = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    finally:
        if args.output:
            stream.close()


if __name__ == "__main__":
    main()
