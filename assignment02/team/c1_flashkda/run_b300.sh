#!/usr/bin/env bash
set -euo pipefail

# One-command baseline/NCU data collection for C1. The vendored snapshot intentionally
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
MICRO_DIR=$(mktemp -d /tmp/flashkda-c1-micro.XXXXXX)
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
    "$PYTHON" benchmarks/bench_fwd.py --H 12 --D 128 \
        --warmup 30 --iters 200 --repeats 5 | tee "$RESULTS/benchmark_h12.txt"
    "$PYTHON" -m pytest -q \
        tests/test_fwd.py::test_fwd tests/test_fwd.py::test_fwd_varlen \
        | tee "$RESULTS/correctness.txt"

    PYTHONPATH="$FLASH_KDA_ROOT/tests:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" "$SCRIPT_DIR/experiments/validate_kernel_matched.py" \
        --flash-root "$FLASH_KDA_ROOT" | tee "$RESULTS/exact_matrix.txt"
    FLA_FLASH_KDA=0 PYTHONPATH="$FLASH_KDA_ROOT/tests:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" "$SCRIPT_DIR/experiments/validate_fla_refs.py" \
        | tee "$RESULTS/fla_refs.txt"
    for HEADS in 96 64 12; do
        FLA_FLASH_KDA=0 PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON" "$SCRIPT_DIR/experiments/bench_c1.py" --heads "$HEADS" \
            --warmup 30 --iters 200 --repeats 5 \
            | tee "$RESULTS/fair_benchmark_h${HEADS}.txt"
    done

    EXTENSION=$("$PYTHON" -c 'import torch, flash_kda_C; print(flash_kda_C.__file__)')
    "$SCRIPT_DIR/experiments/extract_sass.sh" "$EXTENSION" "$RESULTS/flash_kda.sass" \
        | tee "$RESULTS/sass_matrix_opcode_count.txt"
    cuobjdump --dump-ptx "$EXTENSION" >"$RESULTS/flash_kda.ptx" || true

    for CASE in fixed varlen_8x1024; do
        ncu --set full --kernel-name-base function \
            -k 'regex:_flash_kda_fwd_(prepare|recurrence)' \
            --clock-control none --import-source yes --source-folders . \
            --export "$RESULTS/${CASE}_h96" \
            "$PYTHON" "$SCRIPT_DIR/experiments/profile_flash_kda.py" \
                --case "$CASE" --heads 96
        ncu --import "$RESULTS/${CASE}_h96.ncu-rep" --page raw --csv \
            >"$RESULTS/${CASE}_h96_metrics.csv"
    done
)

nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a \
    -o "$MICRO_DIR/mma16_vs_tcgen05" "$SCRIPT_DIR/experiments/mma16_vs_tcgen05.cu"
for INNER in 1 8 64 4096; do
    "$MICRO_DIR/mma16_vs_tcgen05" "$INNER" 200
done | tee "$RESULTS/mma16_vs_tcgen05.txt"

nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a \
    -o "$MICRO_DIR/state_delta_tcgen05" "$SCRIPT_DIR/experiments/state_delta_tcgen05.cu"
for INNER in 1 8 64 256; do
    "$MICRO_DIR/state_delta_tcgen05" "$INNER" 200
done | tee "$RESULTS/state_delta_tcgen05.txt"

"$PYTHON" "$SCRIPT_DIR/experiments/chunk_model.py" \
    | tee "$RESULTS/chunk_model.csv"
"$PYTHON" "$SCRIPT_DIR/experiments/state_precision.py" \
    | tee "$RESULTS/state_precision.csv"

echo "results: $RESULTS"
