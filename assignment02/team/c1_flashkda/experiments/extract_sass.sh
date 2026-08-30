#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <CUDA binary-or-extension> <output.sass>" >&2
    exit 2
fi

BINARY=$(realpath "$1")
OUTPUT=$(realpath -m "$2")
TMP=$(mktemp -d /tmp/flashkda-sass.XXXXXX)
cleanup() { rm -rf -- "$TMP"; }
trap cleanup EXIT

(
    cd "$TMP"
    cuobjdump -xelf all "$BINARY" >/dev/null
    shopt -s nullglob
    CUBINS=(*.cubin)
    if [[ ${#CUBINS[@]} -eq 0 ]]; then
        echo "no cubin extracted from $BINARY" >&2
        exit 1
    fi
    for CUBIN in "${CUBINS[@]}"; do
        echo "===== $CUBIN ====="
        nvdisasm "$CUBIN"
    done
) >"$OUTPUT"

printf 'HMMA.16816 %s\n' "$(grep -c 'HMMA\.16816' "$OUTPUT" || true)"
printf 'UTCHMMA   %s\n' "$(grep -c 'UTCHMMA' "$OUTPUT" || true)"
