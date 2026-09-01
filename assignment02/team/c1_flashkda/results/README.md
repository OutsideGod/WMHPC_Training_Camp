# C1 B300 result index

All authoritative measurements used by `REPORT.md` were collected on one
NVIDIA B300 SXM6 AC (SM103), CUDA 13.0, PyTorch 2.13.0+cu130.

| Result directory | Authoritative contents |
|---|---|
| `job_12114/` | Official H96/H64 benchmark, 1000 timing samples per row |
| `job_12169/` | Baseline fixed/varlen exact-match against kernel-matched `torch_ref` (bf16 state) |
| `job_12186/` | Baseline SASS, fixed/8x1024 NCU reports, microbenchmarks, analytical models |
| `job_12207/` | Successful V-split build, flag check, bf16-state exact-match, benchmark, SASS count |
| `job_13306/` | Final clean default/V-split builds; official + 20-case exact matrices; naive/chunk references; H96/H64/H12 paired benchmarks; corrected microbenchmarks |

The job 12207 challenge build reports `k2_vsplit_enabled True`. Both fixed and
mixed-varlen bf16-state tests have zero output error and exact final-state equality.
Performance is a negative result: the correctness-first scalar half-state and
half-output epilogues dominate, so the experimental branch is not dispatched
by default.

`torch_ref` is deliberately kernel-matched (inline CUDA gate approximation and
cuBLAS fp16-acc GEMM), so its exact equality checks implementation fidelity,
not independent KDA semantics or high-precision accuracy. Those are reported
separately against vendored `fla_kda_ref/naive.py` and `chunk.py`.

Job 13306 is the final correctness authority. Both compile flags pass the two
official H96 tests, all 20 representative exact-match cases (including fp32,
one-sided/no-state, B>1, tails, H12 and varlen), and four independent-reference
checks below the predeclared 1% relative-RMS threshold. The challenge remains a
performance negative result at H12 as well: default/V-split is 0.345x fixed,
0.344x mixed-varlen and 0.305x for 8x1024.
