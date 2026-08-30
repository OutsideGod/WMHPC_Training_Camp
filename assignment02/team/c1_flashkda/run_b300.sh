#!/usr/bin/env bash
set -euo pipefail

# One-command B300 data collection for C1.  The vendored snapshot intentionally
# omits CUTLASS; point FLASH_KDA_ROOT at the complete pinned clone when needed:
#   FLASH_KDA_ROOT=/path/to/FlashKDA bash run_b300.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FLASH_KDA_ROOT=${FLASH_KDA_ROOT:-"$SCRIPT_DIR/FlashKDA"}
PYTHON=${PYTHON:-python}

if [[ ! -d "$FLASH_KDA_ROOT/cutlass/include" ]]; then
    echo "CUTLASS is missing under: $FLASH_KDA_ROOT" >&2
    echo "Use the complete FlashKDA clone pinned to 1ce47ea (CUTLASS 5c149f5)," >&2
    echo "then set FLASH_KDA_ROOT=/path/to/that/clone." >&2
    exit 2
fi
if ! command -v ncu >/dev/null; then
    echo "ncu is required" >&2
    exit 2
fi
if ! command -v nvcc >/dev/null; then
    echo "nvcc is required" >&2
    exit 2
fi
if ! "$PYTHON" -c 'import fla' >/dev/null 2>&1; then
    echo "flash-linear-attention >= 0.5.0 is required by the official benchmark" >&2
    echo "Install it in the selected PYTHON environment before running this script." >&2
    exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS="$SCRIPT_DIR/results/$STAMP"
mkdir -p "$RESULTS"

"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "no visible CUDA GPU"
p = torch.cuda.get_device_properties(0)
print(f"device={p.name} capability={p.major}.{p.minor} memory={p.total_memory}")
assert p.major == 10, "C1 collection must run on an SM100-family GPU"
PY

(
    cd "$FLASH_KDA_ROOT"
    "$PYTHON" -m pip install -v --no-build-isolation -e .
    "$PYTHON" benchmarks/bench_fwd.py --H 96 --D 128 \
        --warmup 30 --iters 200 --repeats 5 | tee "$RESULTS/benchmark_h96.txt"
    "$PYTHON" benchmarks/bench_fwd.py --H 64 --D 128 \
        --warmup 30 --iters 200 --repeats 5 | tee "$RESULTS/benchmark_h64.txt"
    "$PYTHON" -m pytest -q \
        tests/test_fwd.py::test_fwd tests/test_fwd.py::test_fwd_varlen \
        | tee "$RESULTS/correctness.txt"

    EXTENSION=$("$PYTHON" -c 'import torch, flash_kda_C; print(flash_kda_C.__file__)')
    "$SCRIPT_DIR/experiments/extract_sass.sh" "$EXTENSION" "$RESULTS/flash_kda.sass" \
        | tee "$RESULTS/sass_matrix_opcode_count.txt"
    cuobjdump --dump-ptx "$EXTENSION" >"$RESULTS/flash_kda.ptx" || true

    ncu --set full --kernel-name-base function \
        -k 'regex:_flash_kda_fwd_(prepare|recurrence)' \
        --clock-control none --import-source yes --source-folders . \
        --export "$RESULTS/fixed_h96" \
        "$PYTHON" benchmarks/bench_fwd.py --mode fixed --H 96 --D 128 \
            --warmup 0 --iters 1 --repeats 1
    ncu --import "$RESULTS/fixed_h96.ncu-rep" --page raw --csv \
        >"$RESULTS/fixed_h96_metrics.csv"
)

nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a \
    -o "$RESULTS/mma16_vs_tcgen05" "$SCRIPT_DIR/experiments/mma16_vs_tcgen05.cu"
for INNER in 1 8 64 4096; do
    "$RESULTS/mma16_vs_tcgen05" "$INNER" 200
done | tee "$RESULTS/mma16_vs_tcgen05.txt"

nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a \
    -o "$RESULTS/state_delta_tcgen05" "$SCRIPT_DIR/experiments/state_delta_tcgen05.cu"
for INNER in 1 8 64 256; do
    "$RESULTS/state_delta_tcgen05" "$INNER" 200
done | tee "$RESULTS/state_delta_tcgen05.txt"

"$PYTHON" "$SCRIPT_DIR/experiments/chunk_model.py" \
    | tee "$RESULTS/chunk_model.csv"
"$PYTHON" "$SCRIPT_DIR/experiments/state_precision.py" \
    | tee "$RESULTS/state_precision.csv"

echo "results: $RESULTS"
