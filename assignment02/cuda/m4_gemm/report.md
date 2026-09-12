prob 4.1

把3.2的单tile扩成grid覆盖整个矩阵，再沿K每64个元素循环，每轮由普通线程把A、B搬到smem，然后发射四条k16 tcgen05.mma

4096 * 4096 * 4096逐元素判测PASS，性能为41.1TFLOPS，只有cuBLAS的2%

NCU中DRAM利用率0.32%，SM利用率30.01%，说明不是显存带宽打满，而是普通线程搬运、地址计算、shared store和同步占用了大量时间


prob 4.2

TMA把global load、地址计算、swizzle和shared store从普通线程手里接走，只需要elect线程发出两条copy命令，再用mbarrier等待完成

4096 * 4096 * 4096为571.1TFLOPS，是4.1的13.9倍，正确性PASS

此时只有一个buffer，必须先等TMA再做mma，mma结束后才能覆盖smem，所以两者还是串行的


prob 4.3

1. stage扫描结果，顺序是S=2/3/4/6，单位为TFLOPS

4096 * 4096 * 4096：586.8/616.5/535.8/322.5
256 * 4096 * 16384：232.8/317.5/335.5/329.0

八组全部PASS，4096方阵使用S=3时为619.3TFLOPS，达到cuBLAS的35%

2. 每个stage需要24576B smem，S=2/3/4/6都只能常驻1个block
大grid本身已经有足够多的block，S=3以后继续增加stage只会增加buffer和barrier开销
小grid的K循环更长，block间并发少，因此需要更多stage在block内部隐藏延迟，S=4最好

3. S=3的稳态

时间      t0       t1       t2       t3
stage0   TMA0     MMA0     TMA3     MMA3
stage1   TMA1     等待     MMA1     TMA4
stage2   TMA2     等待     等待     MMA2

4. 4.1主要卡普通线程搬运和同步，4.2改成TMA后主要卡单buffer串行，4.3重叠搬运和计算后主要卡tile结构、epilogue和block调度

5. naive到tiled增加了Tensor Core和片上复用，tiled到TMA减少了普通线程搬运，TMA到pipeline隐藏了等待时间

6. 当前TMEM只占64列，而每增加一个stage就增加约24KB smem，因此继续扩大时smem先成为限制，cta_group::2可以通过减少B的smem稍微推迟这个限制


prob 4.5

B300的机器平衡点为2250TFLOPS / 8000GB/s=281.25FLOP/B

1. 七个投影层全部完成测试，M<=16时性能明显下降，最高只有98TFLOPS，M从64到1024快速上升，普通K形状在M>=4096后稳定在约1100-1311TFLOPS

2. M<=16时AI约等于M，远低于281.25，主要在memory roof一侧
以in_proj_qkvgfab为例，M=1/16/256/1024时AI为1.0/15.9/237.8/784.2，实测为4.6/71.8/851.7/1150.9TFLOPS

3. f_b_proj的K只有128，AI最大也只有约118FLOP/B，同时K循环过短，TMA和Tensor Core的准备成本无法摊薄，因此计算和显存带宽都没有打满

4. M<=16时权重几乎没有复用，Tensor Core的tile和TMA准备成本太大，CUDA Core skinny kernel结构更简单，因此vLLM在这里不用Tensor Core
