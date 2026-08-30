# C1 B300 result index

All authoritative measurements used by `REPORT.md` were collected on one
NVIDIA B300 SXM6 AC (SM103), CUDA 13.0, PyTorch 2.13.0+cu130.

| Result directory | Authoritative contents |
|---|---|
| `job_12114/` | Official H96/H64 benchmark, 1000 timing samples per row |
| `job_12169/` | Baseline fixed/varlen exact-match against `torch_ref` |
| `job_12186/` | Baseline SASS, fixed/8x1024 NCU reports, microbenchmarks, analytical models |
| `job_12207/` | Successful V-split build, flag check, exact-match, benchmark, SASS |

The final challenge build reports `k2_vsplit_enabled True`. Both fixed and
mixed-varlen tests have zero output error and exact final-state equality.
Performance is a negative result: the correctness-first scalar half-state and
half-output epilogues dominate, so the experimental branch is not dispatched
by default.
