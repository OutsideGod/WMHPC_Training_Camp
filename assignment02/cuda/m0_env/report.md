prob 0.1:

现象：
CUDA error cudaErrorNoKernelImageForDevice at m0_env/01_first_mma.cu:76: no kernel image is available for execution on the device
make: *** [Makefile:42: run/m0_env/01_first_mma] Error 1
rm bin/m0_env/01_first_mma

失败原因：
只编译了sm80版本的cubin，计算能力不跨major版本兼容，且在此命令下fatbin中不生成PTX，因此计算能力为12.x的5090/B300无法兼容


prob 0.2:

标准：
dense、boost/标称峰值频率、单 GPU、FMA = 2 FLOP。
BF16 采用 FP32 accumulate

这题其实出的有问题，因为架构的迭代，5090和B300矩阵乘法使用的是比mma.sync先进得多的tcgen05，无法轻易的通过warp级别的运算换算整体的速率，因此白皮书上并不着重视hmma多少的延迟多少cycle之类的数据。为了真实性我将放弃这个意义不大的计算，直接采用上面对5090和B300的bf16真实峰值，在此基础上算其他数值。
5090：209.5 TFLOPS、419 TFLOPS、838 TFLOPS、116.9FLOP/Byte = 209.5TFLOPS / 1792GB/s
B300：2250 TFLOPS、4500 TFLOPS、9000 TFLOPS、281.25 FLOP/Byte = 2250 TFLOPS / 8000 GB/s
116.9、281.25比3.6大了大概两个数量级，Tensor Core的计算能力已经远远快于显存直接供给数据的能力


prob 0.3：

1. 正确

2. 正确

3. 错误，算力、存储上都有代价，指令形状过大会导致内存不足，且整个任务在单个大矩阵上完成会导致需要传完数据再进行计算，利用率低

4. 错误，如同上面prob 0.2的结果，数据复用可以增大效率，使得机器平衡高于单条mma平衡
