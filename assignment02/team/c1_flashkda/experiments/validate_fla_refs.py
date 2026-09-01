#!/usr/bin/env python3
"""Independent C1 correctness checks against vendored FLA references.

The pure-PyTorch recurrent reference covers small fixed/varlen cases.  The
vendored Triton ``chunk.py`` covers larger cases with ``safe_gate=True``.
``FLA_FLASH_KDA=0`` is set before importing FLA so the comparison cannot
silently dispatch back into the kernel under test.
"""

from __future__ import annotations

import argparse
import math
import os

os.environ["FLA_FLASH_KDA"] = "0"

import torch

import flash_kda
from fla_kda_ref import chunk_kda
from fla_kda_ref.gate import naive_kda_lowerbound_gate
from fla_kda_ref.naive import naive_recurrent_kda


def rel_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.double() - expected.double()).square().mean().sqrt()
    base = expected.double().square().mean().sqrt().clamp_min(1e-30)
    return (diff / base).item()


def prefix_tensor(seq_lens: list[int]) -> torch.Tensor | None:
    if len(seq_lens) == 1:
        return None
    values = [0]
    for length in seq_lens:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int64, device="cuda")


def make_inputs(seq_lens: list[int], heads: int, seed: int):
    torch.manual_seed(seed)
    total = sum(seq_lens)
    shape = (1, total, heads, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(heads, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn((heads, 128), dtype=torch.float32, device="cuda")
    h0 = torch.randn((len(seq_lens), heads, 128, 128), dtype=torch.float32, device="cuda")
    return q, k, v, g, beta, a_log, dt_bias, h0


def run_flash(seq_lens, tensors):
    q, k, v, g, beta, a_log, dt_bias, h0 = tensors
    out = torch.empty_like(v)
    final = torch.empty_like(h0)
    cu = prefix_tensor(seq_lens)
    kwargs = {"cu_seqlens": cu} if cu is not None else {}
    flash_kda.fwd(
        q, k, v, g, beta, 1.0 / math.sqrt(128), out,
        A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
        initial_state=h0, final_state=final, **kwargs,
    )
    return out, final


def run_naive(seq_lens, tensors):
    q, k, v, g, beta, a_log, dt_bias, h0 = tensors
    # Match fused input transforms, but keep the recurrence itself independent.
    qn = (q.float() / torch.sqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
    kn = (k.float() / torch.sqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
    gate = naive_kda_lowerbound_gate(g, a_log, dt_bias, lower_bound=-5.0)
    beta_act = beta.float().sigmoid()

    outputs = []
    states = []
    begin = 0
    for index, length in enumerate(seq_lens):
        end = begin + length
        out, state_kv = naive_recurrent_kda(
            qn[:, begin:end], kn[:, begin:end], v[:, begin:end],
            gate[:, begin:end], beta_act[:, begin:end],
            scale=1.0 / math.sqrt(128),
            initial_state=h0[index:index + 1].transpose(-2, -1),
            output_final_state=True,
        )
        outputs.append(out)
        states.append(state_kv.transpose(-2, -1))
        begin = end
    return torch.cat(outputs, dim=1), torch.cat(states, dim=0)


def run_chunk(seq_lens, tensors):
    q, k, v, g, beta, a_log, dt_bias, h0 = tensors
    cu = prefix_tensor(seq_lens)
    kwargs = {"cu_seqlens": cu} if cu is not None else {}
    return chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=1.0 / math.sqrt(128),
        initial_state=h0,
        output_final_state=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        state_v_first=True,
        **kwargs,
    )


def check(name, flash, reference, threshold):
    out_err = rel_rms(flash[0], reference[0])
    state_err = rel_rms(flash[1], reference[1])
    print(
        f"PASS reference={name} output_rel_rms={out_err:.6e} "
        f"state_rel_rms={state_err:.6e} threshold={threshold:.3e}"
    )
    if out_err > threshold or state_err > threshold:
        raise AssertionError(
            f"{name} exceeds threshold: output={out_err:.6e} state={state_err:.6e}"
        )


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--threshold", type=float, default=0.01)
    args = p.parse_args()

    print("FLA_FLASH_KDA=0")
    print("chunk_reference=fla_kda_ref/chunk.py safe_gate=True")
    print("naive_reference=fla_kda_ref/naive.py pure PyTorch recurrence")

    for index, seq_lens in enumerate(([31], [17, 33])):
        tensors = make_inputs(list(seq_lens), heads=1, seed=args.seed + index)
        flash = run_flash(list(seq_lens), tensors)
        check(f"naive_{'varlen' if len(seq_lens) > 1 else 'fixed'}", flash,
              run_naive(list(seq_lens), tensors), args.threshold)

    for index, (seq_lens, heads) in enumerate((([257], 4), ([33, 65, 129], 4))):
        tensors = make_inputs(seq_lens, heads=heads, seed=args.seed + 10 + index)
        flash = run_flash(seq_lens, tensors)
        check(f"chunk_{'varlen' if len(seq_lens) > 1 else 'fixed'}", flash,
              run_chunk(seq_lens, tensors), args.threshold)

    print("SUMMARY independent_reference_cases=4 status=PASS")


if __name__ == "__main__":
    main()
