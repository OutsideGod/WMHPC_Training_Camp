from pathlib import Path
import sys

import tilelang
from tilelang import tvm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "assignment01"))
from kernels.tilelang_matmul import make_matmul


def main():
    out = Path(__file__).resolve().parent / "generated"
    out.mkdir(exist_ok=True)
    config = dict(M=128, N=128, K=64, BLOCK_M=128, BLOCK_N=128,
                  BLOCK_K=32, threads=128, num_stages=3)

    for arch in ("sm_90a", "sm_100a"):
        target = tvm.target.Target({"kind": "cuda", "arch": arch})
        with target:
            func = make_matmul(**config)
            artifact = tilelang.lower(func, target=target)
            tilelang.compile(make_matmul(**config), out_idx=[2], target=target)
        (out / f"{arch}.cu").write_text(artifact.kernel_source)
        lowering = str(artifact.host_mod) + "\n\n" + str(artifact.device_mod)
        (out / f"{arch}.tir").write_text(lowering)
        source = artifact.kernel_source
        selected = (
            "wgmma" if "wgmma" in source else
            "tcgen05" if "tcgen05" in source else
            "mma_sync" if "mma_sync" in source else "unknown"
        )
        print(f"{arch}: compile PASS, Tensor Core path={selected}, "
              f"CUDA={len(source)} bytes")


if __name__ == "__main__":
    main()
