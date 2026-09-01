#!/usr/bin/env python3
"""Focused exact-match matrix for the final C1 FlashKDA build.

This validates implementation fidelity against FlashKDA's kernel-matched
``torch_ref``.  It is deliberately separate from ``validate_fla_refs.py``:
exact equality here catches indexing/layout/template mistakes, while the FLA
references independently check KDA semantics with numerical tolerances.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    seq_lens: tuple[int, ...]
    heads: int


CASES = (
    Case("tail_T1_H1", 1, (1,), 1),
    Case("tail_T15_H4", 1, (15,), 4),
    Case("boundary_T16_H4", 1, (16,), 4),
    Case("tail_T17_H12", 1, (17,), 12),
    Case("tail_T33_H64", 1, (33,), 64),
    Case("small_H96", 1, (17,), 96),
    Case("batched_B2_T17_H12", 2, (17, 17), 12),
    Case("varlen_edges_H12", 1, (1, 15, 16, 17, 33), 12),
)

STATE_MATRIX = (
    ("bf16_in_out", torch.bfloat16, True, True),
    ("fp32_in_out", torch.float32, True, True),
    ("bf16_in_only", torch.bfloat16, True, False),
    ("fp32_in_only", torch.float32, True, False),
    ("bf16_out_only", torch.bfloat16, False, True),
    ("fp32_out_only", torch.float32, False, True),
    ("no_state", None, False, False),
)


def make_inputs(batch: int, total: int, heads: int, seed: int):
    torch.manual_seed(seed)
    shape = (batch, total // batch, heads, 128)
    q = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(heads, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand((heads, 128), dtype=torch.float32, device="cuda")
    return q, k, v, g, beta, a_log, dt_bias


def run_one(case: Case, state_spec, torch_ref, flash_kda, seed: int) -> None:
    state_name, state_dtype, has_in, has_out = state_spec
    total = sum(case.seq_lens)
    q, k, v, g, beta, a_log, dt_bias = make_inputs(case.batch, total, case.heads, seed)
    n_seq = len(case.seq_lens)
    cu_seqlens = None
    if case.batch == 1 and n_seq > 1:
        prefix = [0]
        for length in case.seq_lens:
            prefix.append(prefix[-1] + length)
        cu_seqlens = torch.tensor(prefix, dtype=torch.int64, device="cuda")

    initial_kernel = initial_ref = None
    if has_in:
        base = torch.randn((n_seq, case.heads, 128, 128), dtype=torch.float32, device="cuda")
        initial_kernel = base.to(state_dtype)
        initial_ref = initial_kernel.clone()
    final_kernel = final_ref = None
    if has_out:
        final_kernel = torch.empty(
            (n_seq, case.heads, 128, 128), dtype=state_dtype, device="cuda"
        )
        final_ref = torch.empty_like(final_kernel)

    out_kernel = torch.empty_like(q)
    out_ref = torch.empty_like(q)
    kwargs = {"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}
    scale = 1.0 / math.sqrt(128)
    flash_kda.fwd(
        q, k, v, g, beta, scale, out_kernel,
        A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
        initial_state=initial_kernel, final_state=final_kernel, **kwargs,
    )
    torch_ref(
        q, k, v, g, beta, scale, out_ref,
        A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
        initial_state=initial_ref, final_state=final_ref, **kwargs,
    )
    torch.cuda.synchronize()

    if not torch.equal(out_kernel, out_ref):
        diff = (out_kernel.float() - out_ref.float()).abs()
        raise AssertionError(
            f"{case.name}/{state_name}: output mismatch "
            f"max={diff.max().item():.6g} mean={diff.mean().item():.6g}"
        )
    if has_out and not torch.equal(final_kernel, final_ref):
        diff = (final_kernel.float() - final_ref.float()).abs()
        raise AssertionError(
            f"{case.name}/{state_name}: state mismatch "
            f"max={diff.max().item():.6g} mean={diff.mean().item():.6g}"
        )
    print(f"PASS exact case={case.name} state={state_name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flash-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260831)
    args = p.parse_args()

    tests_dir = args.flash_root.resolve() / "tests"
    if not (tests_dir / "torch_ref.py").is_file():
        p.error(f"torch_ref.py not found under {tests_dir}")
    sys.path.insert(0, str(tests_dir))

    import flash_kda
    import flash_kda_C
    from torch_ref import torch_ref

    get_flag = getattr(flash_kda_C, "k2_vsplit_enabled", None)
    flag = get_flag() if get_flag is not None else "upstream-uninstrumented"
    print(f"k2_vsplit_enabled={flag}")
    print("reference=torch_ref (kernel-matched; exact equality)")

    count = 0
    # Every shape gets a bf16 input/output state check.
    for index, case in enumerate(CASES):
        run_one(case, STATE_MATRIX[0], torch_ref, flash_kda, args.seed + index)
        count += 1

    # Exercise every state template on both a fixed and a varlen tail case.
    for case in (CASES[3], CASES[-1]):
        for index, state_spec in enumerate(STATE_MATRIX[1:], start=1):
            run_one(case, state_spec, torch_ref, flash_kda, args.seed + 100 + index)
            count += 1

    print(f"SUMMARY exact_cases={count} status=PASS")


if __name__ == "__main__":
    main()
