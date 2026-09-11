prob 2.1

1. st.shared / fence.proxy.async / wgmma.fence / wgmma.mma_async / wgmma.commit_group / wgmma.wait_group

fence.proxy.async：避免SMEM写入和mma乱序
wgmma.fence：避免对RMEM读写和mma乱序
wgmma.commit_group和wgmma.wait_group：避免mma和对RMEM读写乱序

2.错，只要涉及proxy内存可见性不同都得用
错，commit只管发射，wait才会等待
对


prob 2.2

本质区别在于数据存储顺序不同，是沿着对应major，以atom为单位展开成一维存储
实际体现区别在于在两个major对应逻辑矩阵的元素数量不同时，lbo不变，但sbo会不同，且比值等于对应的元素数量比，比如m * k=128 * 64，那mmajor和kmajor的sbo比为2 : 1
在题目的例子中，逻辑矩阵为64 * 64，此时两major元素数量相等，因此sbo、lbo数值均相等