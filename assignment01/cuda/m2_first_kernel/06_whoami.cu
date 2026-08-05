// 问题 2.8：观察执行顺序。
// 连续运行两三次，对比 block 出现的先后顺序，回答 handout 里的问题。
#include "common.h"

__global__ void whoami() {
    // 让每个 block 的 0 号线程报到。
    if (threadIdx.x == 0) {
        printf("block %d 报到\n", blockIdx.x);
    }
}

int main() {
    whoami<<<16, 32>>>();
    CUDA_CHECK_KERNEL();
    return 0;
}
//顺序由硬件调度器决定，不可以依赖顺序，Block之间互不依赖且无序执行