prob 6.1

使用assignment01/kernels/tilelang_matmul.py，把同一个T.gemm分别编译为sm_90a和sm_100a，两边compile全部PASS

sm_90a：wgmma m64n128k16，lowering生成TMA和wgmma descriptor，根据tile和wgmma决定smem swizzle，T.copy被lowering成TMA

sm_100a：mma.sync m16n8k16，lowering生成TMA descriptor，mma.sync使用ldmatrix供数，根据tile和ldmatrix决定smem swizzle，T.copy被lowering成TMA

sm_100a实际生成的是mma.sync而不是tcgen05，说明当前tilelang 0.1.13还不会把这个T.gemm自动降成tcgen05

1. 编译器自动完成Tensor Core指令、线程映射、smem布局、swizzle、descriptor、TMA和流水同步

2. 程序员仍然需要决定BLOCK_M、BLOCK_N、BLOCK_K、threads、num_stages和精度，编译器不会自动保证这些参数最快
