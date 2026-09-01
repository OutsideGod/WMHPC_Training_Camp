# C1 答辩稿（10 分钟）

> 官方 GB200 与本组 B300 数据分开标注；原始日志均保存在 `results/`。

## 1. 问题与结论（0:00--0:50）

- 问题：FlashKDA 已在 Blackwell 上很快，但矩阵乘仍用 m16n8k16 SM80 MMA；
  是否值得做 tcgen05 专版？
- 结论：当前证据不支持全量替换或 dispatch 现有 V-split；后者 fixed H96 只有
  0.363×，修正计时口径后的 state-delta tcgen05 单次微基准也只有 0.455×。
  这是当前原型的 kill decision，不是“所有 SM100 方案不可能加速”的定理。
- 发布门槛：fixed/TP 形状至少 +10%，varlen 回退不超过 3%，精度不退。

## 2. 先看数据流，不看架构标签（0:50--1:50）

```text
K1: N×H×chunks CTAs
q/k/g → decay + L/Mqk + inverse → 648 MiB workspace
                                      ↓
K2: N×H CTAs, each CTA serially walks all chunks
v + workspace + recurrent state → out/state
```

- K1 token-parallel，K2 只有 head/sequence 并行。
- K2 用 `MOVM_T` 将 U 留在寄存器，连续喂给后续 16×16 MMA。
- 官方曾融合 K1/K2，因 K1 并行度被压低而至少慢 15%。

token 级 V-first recurrence 是：

```text
S_decay = S_(t-1) diag(exp(g_t))
e_t     = v_t - S_decay k_t
S_t     = S_decay + beta_t e_t k_t^T
o_t     = S_t (scale q_t)
```

沿 V 行切成两个 64-row block 后，上述四步都逐 block 闭合，只共享只读的
`q/k/g/beta`；沿 K 切则需要归约。这是 V-split 无跨 CTA 同步的代数依据。

## 3. 复现与指令证据（1:50--2:40）

- 源码唯一 MMA atom：`SM80_16x8x16_*`；TMA/STSM 只是供数路径。
- B300 SASS：`HMMA.16816` 1544 处；`UTCHMMA` 0 处；现场展示提交中保留的
  opcode count/sample，完整约 27 MiB SASS 由脚本再生。
- B300 H=96 fixed：0.9999 ms；8×1024 varlen：0.7327 ms（均取 fp32 state）。
- fixed ncu K1/K2：273.856 / 719.008 us，K2 占 72.4%。

展示 `results/<job>/default_sass_opcode_{count,sample}.txt` 和 benchmark 表，
不展示整页日志或可再生的 27 MiB SASS。

## 4. CHUNK=32 为什么不能只改常量（2:40--4:00）

| C | worst `exp(+5C)` | worst bf16 `exp(-5C)` | FLOP/token | workspace |
|---:|---:|---:|---:|---:|
| 16 | 5.54e34 | 1.81e-35 | 117,760 | 648 MiB |
| 32 | inf | 0 | 147,456 | 684 MiB |
| 64 | inf | 0 | 245,760 | 774 MiB |

- 第一处硬失败是数值范围：C=32 已 overflow/underflow。
- 然后才是 inverse：6→8→10 次 C³ GEMM。
- 大 chunk 必须配 rescale；即便修数值，C=32/64 的 GEMM 工作仍 +25%/+109%。

## 5. tcgen05 最小 tile 的有效利用率（4:00--5:20）

| 阶段 | M | 最小 M=64 的行利用率 |
|---|---:|---:|
| L/Mqk/inverse | 16 | 25% |
| k/q × state 合并 | 32 | 50% |
| state delta | 128 | 100% |

- tcgen 的峰值是 physical FLOP/s；我们关心 useful FLOP/s。
- 结果在 TMEM，需 alloc/commit/mbarrier/`tcgen05.ld`；会打断当前寄存器融合。
- 单次 m16 microbench：mma.sync 4.117 us；tcgen padded 6.156 us；0.669×。
- 单次 state-delta m128n128k16：4-warp MMA 10.253 us，tcgen 22.532 us，0.455×；
  只有人为重复 256 次摊薄 setup 后 tcgen 才达到 2.358×。
- 这两个数字是单 CTA microbench 端点：重复使用同一 A/B、同一 accumulator 且只
  drain 一次。真实 K2 可把 alloc 移出循环，但每 chunk 会换输入并递推 state，不能
  把任一端点直接当端到端预测。

## 6. 真正瓶颈：并行度与数据移动（5:20--6:40）

- useful AI 约 39 FLOP/B；B300 平衡点约 281 FLOP/B，不在 compute side。
- fixed K2 ncu：DRAM 19.50% peak、tensor pipe 20.14%、active warps 9.37%、
  waves/SM 仅 0.32；8×1024 时 waves/SM 2.59，K2 降到 416.608 us。
- 官方 GB200 同 token：拆成 8 个 sequence 后 H96 +1.43×、H64 +1.92×。
- 解释：K2 blocks 从 H 增至 8H，串行链从 512 chunks 缩至 64。
- 公平 FLA 对照显式 `safe_gate=True, FLA_FLASH_KDA=0`：H12 fixed 时 FlashKDA
  1.3630 ms、FLA 0.8642 ms（仅 0.63×）；8×1024 时 FlashKDA 才以 1.92× 领先。
- TP8 每卡只有 12 heads，是最该优化而非最该减少 CTA 的场景。

## 7. 候选方案与反例（6:40--7:50）

- 多 head/CTA：不同 head 不共享 state/B，且约 98 KiB smem/head；拒绝。
- persistent：K2 已 persistent；不算新方案。
- K1+K2 fuse：官方负结果；拒绝重复。
- 双 CTA/head 的 V-split：每 CTA 负责 64 个 V 行且各自独立递推，blocks 翻倍；
  默认/挑战各通过官方大形状与 20 项状态/尾块矩阵 exact-match，并通过 naive/chunk
  四项独立参考（output/state rel-RMS 0.44%--0.58%，阈值 1%）。
- 结果：H96 fixed 0.9999→2.7582 ms，8×1024 0.7327→2.8235 ms；完整左操作数
  重复加载与 correctness-first 标量半幅 store 触发 kill criteria。
- 最强反例 H12 也失败：paired default/V-split 为 fixed 0.345×、mixed 0.344×、
  8×1024 0.305×。

## 8. bf16 state 不是靠 exact-match 证明（7:50--8:50）

- 现有 torch_ref 复刻同样 bf16 舍入，只证明 kernel 写对了。
- 3-seed fp64 gold 合成隔离实验：T=8192，弱衰减时 state/probe rel-RMS
  2.298±0.025% / 2.309±0.037%；中等衰减约 0.24--0.25%。
- 结论：误差由有效记忆长度控制；必须补真实 gate 分布、长上下文、跨层 logits。
- 展示按 token window 的误差斜率，而非只报最终 max abs。

## 9. v2 发布决策（8:50--9:35）

- 默认：portable SM80 MMA 路径继续覆盖 SM90/100/103/120。
- 当前不 dispatch 2-CTA；若继续，先补半宽 TMA 与 32-lane coalesced epilogue。
- tcgen05 仅保留研究分支，不全量重写。
- 当前原型已触发 kill criteria：本次数据不支持替换官方 SM80 MMA 默认路径。

## 10. 收尾与提问（9:35--10:00）

一句话：**这个问题不是“新指令是否更快”，而是“16 行递推能否持续给 64/128 行的
异步 Tensor Core 喂满，并且不丢掉已有寄存器融合和并行度”。目前证据回答是否定的。**

备答：若问“打包四个 head”——普通 GEMM 沿 M 打包要求共享 B；各 head 的 state/B
不同。做 block diagonal 会增加 padding；约 98 KiB/head，四份约 392 KiB 工作集
也放不进一个 CTA。
