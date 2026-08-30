#!/usr/bin/env python3
"""Paper model for C1 discussion points 1 and 2.

The model counts only GEMM FLOPs visible in FlashKDA's K1/K2 source.  It is
not a performance prediction; its purpose is to make CHUNK scaling and
tcgen05 padding explicit before measuring on B300.
"""

from __future__ import annotations

import argparse
import math

import torch


def neumann_gemms(chunk: int) -> int:
    if chunk < 2 or chunk & (chunk - 1):
        raise ValueError("CHUNK must be a power of two >= 2")
    # For each L^(2,4,...,CHUNK/2): form the power and multiply INV by it.
    return 2 * (int(math.log2(chunk)) - 1)


def row(chunk: int, d: int, tokens: int, heads: int) -> dict[str, float]:
    inv_gemm = neumann_gemms(chunk)
    inv_flop = inv_gemm * 2 * chunk**3
    k1_flop = 4 * chunk**2 * d + inv_flop  # L, Mqk, inverse
    k2_flop = 6 * chunk * d**2 + 4 * chunk**2 * d
    tile_bytes = 3 * chunk * d * 2 + d * 4 + 2 * chunk**2 * 2
    tiles = math.ceil(tokens / chunk)
    workspace = heads * tiles * tile_bytes

    # Worst case g=-5 per token.  The kernel materializes both exp(cumsum(g))
    # and exp(-cumsum(g)), then casts the former to bf16.
    exponent = 5.0 * chunk
    pos_f32 = torch.exp(torch.tensor(exponent, dtype=torch.float32)).item()
    neg_bf16 = (
        torch.exp(torch.tensor(-exponent, dtype=torch.float32))
        .to(torch.bfloat16)
        .float()
        .item()
    )
    return {
        "chunk": chunk,
        "neumann_gemms": inv_gemm,
        "k1_flop/token": k1_flop / chunk,
        "k2_flop/token": k2_flop / chunk,
        "total_flop/token": (k1_flop + k2_flop) / chunk,
        "workspace_MiB": workspace / 2**20,
        "workspace_B/token/head": tile_bytes / chunk,
        "exp(+5C)_fp32": pos_f32,
        "exp(-5C)_bf16": neg_bf16,
        # cta_group::1 dense bf16 tcgen05 has M>=64.  A lone CHUNK-row
        # operation therefore uses this fraction of its physical rows.
        "tcgen05_row_util": min(chunk, 64) / 64,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--heads", type=int, default=96)
    p.add_argument("--d", type=int, default=128)
    args = p.parse_args()

    rows = [row(c, args.d, args.tokens, args.heads) for c in (16, 32, 64)]
    columns = list(rows[0])
    print(",".join(columns))
    for values in rows:
        fields = []
        for name in columns:
            value = values[name]
            if isinstance(value, float):
                fields.append(f"{value:.9g}")
            else:
                fields.append(str(value))
        print(",".join(fields))

    print("\nNotes:")
    print("- exp(+5C) overflowing or exp(-5C) becoming zero is a hard failure.")
    print("- Workspace is active tile payload; it excludes the aligned prefix buffer")
    print("  and the wrapper's conservative extra-N-tile allocation.")
    print("- FLOPs exclude normalization, gate activation, decay, and copies.")


if __name__ == "__main__":
    main()
