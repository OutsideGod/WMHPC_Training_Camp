prob 3.1

1. 对，tcgen05.ld每个warp只能读自己对应的32行，因此四个warp分别读0-31、32-63、64-95、96-127

2. 对，tcgen05.mma只需要一个elect线程发射，计算由硬件异步完成

3. 错，TMEM不能直接作为TMA的源，需要先tcgen05.ld到寄存器，再写回gmem

4. 对，整个TMEM为128 * 512 * 4=262144B，m128n256的fp32累加器为128 * 256 * 4=131072B，正好占一半

5. 错，commit只负责提交，必须wait以后才能读取结果


prob 3.2

先把A、B从gmem搬到swizzle后的smem，再分配TMEM并初始化mbarrier，之后发射四条k16 tcgen05.mma，等待完成后由四个warp分别读取32行并写回gmem

B300上五个seed全部逐元素相等，PASS

关闭fence.proxy.async以后seed=42仍然PASS，只能说明这次没有碰到错误，普通st.shared和tcgen05使用的proxy不同，不加fence仍然没有可见性保证


prob 3.3

错误版一直使用phase=0，rounds=1时PASS，rounds=2和4时超时

第一轮结束后phase已经从0翻到1，第二轮还等待0就会提前读取仍在写入的TMEM

改成mbar_wait(round & 1)，让phase每轮在0和1之间交替，rounds=1/2/4、seed=42/7全部PASS


prob 3.4

1. cta_group::1中每个CTA保存完整B，需要8192B，cta_group::2只保存一半B，需要4096B，实测总smem/block从24588B降到20492B

2. 两种实现都PASS，时间为12.78us和14.25us，单tile时间无法说明谁更快
NCU中shared store wavefront从778降到650，Tensor Core shared wavefront从384降到320，都是原来的83.3%，因为A没有减少，只有B减半

3. 每个stage节省4KB B smem，可以用来增加stage、扩大tile或者提高常驻block数量

4. cta_group::2依赖thread block cluster和distributed shared memory，更适合片上资源更多、长期处理大矩阵的数据中心卡
