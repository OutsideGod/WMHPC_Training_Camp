# C1 报告：FlashKDA 是否值得做 SM100 专版

## 结论先行

最终结论是：**现有证据不支持把 FlashKDA v2 的主线改写成全量 `tcgen05` 专版；
保留 SM80 MMA 默认路径，也不 dispatch 当前 K2 双 CTA/V-split 原型。** 后者虽在
bf16/fp32、fixed/varlen、尾块与单边 state 矩阵中均与 kernel-matched 参考逐元素
一致，并通过独立 FLA 参考，但 H96 fixed 只有基线的 0.363×，不满足发布门槛；
tcgen05 state-delta 微基准也只在把同一 MMA
人为重复 256 次、摊薄一次性成本后显示吞吐优势。这是否定**当前原型和直接替换
策略**，不是证明所有 SM100 重构都不可能获益。

理由不是“Blackwell Tensor Core 不够快”，而是这个 workload 的形状和数据流
不匹配：

1. `CHUNK=16` 在 `lower_bound=-5` 时已经接近不做 rescale 的数值极限；直接改成
   32 会同时出现 `exp(+160)=inf` 和 bf16 `exp(-160)=0`。
2. SM100 dense bf16 `tcgen05` 的单 CTA 最小 M 是 64。孤立的 16 行运算只有 25%
   有效行；更重要的是结果落在 TMEM，而当前 K2 用寄存器内 `MOVM_T` 串联多个
   16×16 阶段，直接替换会新增 TMEM 分配、提交、等待和 drain。
3. 源码估算的 useful arithmetic intensity 约为 39 FLOP/B，远低于本作业 B300
   的 281.25 FLOP/B 机器平衡点。官方 GB200 固定长度数据只达到约 91.8 useful
   TFLOP/s，却随序列拆分得到 1.43--1.92× 加速；主要矛盾是 K2 的 `N×H` 并行度
   和串行 chunk 依赖，而不是 Tensor Core 峰值。
4. `CHUNK=32/64` 不只是换常量。它们需要 rescale、新的有限 Neumann 展开以及
   全套 shared-memory/register/pipeline 重排；纸面 GEMM FLOP/token 分别增加
   25.2% 和 108.7%。
5. 挑战原型证实“增加 CTA 数”本身不够：完整左操作数重复加载和 correctness-first
   半幅标量 store 把 H96 fixed 的 0.9999 ms 拉到 2.7582 ms。因此半宽 TMA/协作
   epilogue 是 V-split 的前置条件，而不是后续小优化。

本组已通过 Slurm 在一张 NVIDIA B300 SXM6 AC（SM103）上完成官方 benchmark；
下文严格区分“本组 B300 实测”“官方 GB200 数据”和“分析模型”，不把推算伪装
成测量。仓库跟踪的 `assignment02/run.sh` 是最终 Slurm 验证入口：从同一 pin
分别干净编译默认/V-split 两条路径并保存测试、独立参考、benchmark、SASS 与
microbench。[`run_b300.sh`](run_b300.sh) 是已有完整 FlashKDA 环境下的基线/NCU
采集入口；两者用途不同。

## 1. 复现状态与证据链

### 1.1 环境

| 项目 | 实测环境 |
|---|---|
| CUDA toolkit | 13.0 / nvcc 13.0.88 |
| PyTorch | 2.13.0+cu130 |
| CUDA device | NVIDIA B300 SXM6 AC，compute capability 10.3 |
| Nsight Compute | 2025.3.1.0 |
| FlashKDA / CUTLASS | `1ce47ea3` / `5c149f52` |
| 编译目标 | `sm_103a` |
| benchmark / 最终验证 | `results/job_12114/` / `results/job_13306/` |

完整复现需要 FlashKDA `1ce47ea` 和 CUTLASS `5c149f5`。从仓库运行：

```bash
cd assignment02 && sbatch run.sh
```

脚本会创建隔离环境并安装 `flash-linear-attention==0.5.2`；该依赖用于同时测
Triton `chunk_kda`。临时环境的 `bin` 必须加入 `PATH`，否则 PyTorch 的
`load_inline` 找不到已经安装的 `ninja`。

脚本按 Slurm job id 新建结果目录，不覆盖旧数据；它保存 H=96/H=64/H=12 的三组
形状、默认/V-split 两次独立构建、官方大形状 exact-match、状态模板/尾块矩阵、
`naive.py` 与 `chunk.py` 对拍、扩展 SASS 计数以及两项挑战 microbench。已有基线
的 focused NCU 数据由 `run_b300.sh` 调用 `profile_flash_kda.py` 单独采集，避免把
FLA/GDN 或多种 state launch 混进一个 report。

B300 基线已通过 `test_fwd` 与 `test_fwd_varlen`：output 和 bf16 final state 对
kernel-matched `torch_ref` 均逐元素完全相等，fixed/varlen 的 avg/max atol 都为 0。
这里的 `torch_ref` 会调用内联 CUDA `tanh.approx` 与 cuBLAS fp16-acc GEMM，目的
是复刻 kernel 舍入，不是纯 PyTorch 或独立高精度参考。原始记录在
`results/job_12169/baseline_correctness.txt`；独立语义正确性另对拍题目给出的
`fla_kda_ref/naive.py`（纯 PyTorch recurrence）与 `chunk.py`（禁用 FlashKDA
自动派发的 Triton 路径）。

### 1.2 “主路径是 SM80 MMA”的静态证据

- [`utils.cuh`](FlashKDA/csrc/smxx/utils.cuh) 包含
  `cute/arch/mma_sm80.hpp`，所有 GEMM atom 均为
  `SM80_16x8x16_*`；没有 tcgen05/wgmma atom。
- K1 的 `L`、`Mqk` 和 6 次 Neumann GEMM 都调用上述 m16n16 helper。
- K2 创建的唯一 MMA atom也是 `SM80_16x8x16_F32BF16BF16F32_TN`。
- SM90 TMA/STSM 出现在供数和回写路径，不应误判为 WGMMA。要证明“矩阵乘路径”
  仍是 SM80 MMA，本组对 SM103a 扩展逐个提取 cubin 后用 `nvdisasm` 反汇编：
  `HMMA.16816` 静态出现 **1544** 次，`UTCHMMA` 为 **0**。提交中保留可审阅的
  `results/job_12186/baseline_sass_opcode_count.txt` 与
  `baseline_sass_opcode_sample.txt`；约 27 MiB 的完整 `.sass` 是可由脚本再生的
  忽略产物，不作为干净 checkout 中的链接目标。

挑战 microbench 的 PTX 已在本地编译检查：SM80 对照含
`wmma.mma.sync...m16n16k16...bf16`，SM100 对照含
`tcgen05.mma.cta_group::1.kind::f16`。这只证明实验的两条代码路径正确生成，
不是 FlashKDA 本体的 B300 运行证据。

### 1.3 官方 GB200 基线提供的并行度证据

数据来自 [`BENCHMARK_GB200.md`](FlashKDA/BENCHMARK_GB200.md)，不能冒充本组
B300 实测：

| H | 固定 8192 | 8×1024 varlen | 拆分加速 |
|---:|---:|---:|---:|
| 96 | 1.0087 ms | 0.7064 ms | 1.43× |
| 64 | 0.9247 ms | 0.4811 ms | 1.92× |

本组 B300 的 1000 次计时样本均值如下。为复现官方结果生成器，FlashKDA 取
`fp32 state` 行；括号内是相对该**官方协议**中 FLA `chunk_kda` 的比值：

| H | fixed 8192 | mixed varlen | 8×1024 varlen |
|---:|---:|---:|---:|
| 96 | 0.9999 ms（2.37×） | 0.8721 ms（2.74×） | 0.7327 ms（3.20×） |
| 64 | 0.9099 ms（1.81×） | 0.6592 ms（2.54×） | 0.4947 ms（3.17×） |

原始日志在 `results/job_12114/benchmark_h96.txt` 和 `benchmark_h64.txt`。
该官方脚本没有传 `safe_gate=True`，因此 FLA 走的不是其针对 bounded gate 的
M=16 Tensor Core 快路径；上表的 1.81--3.20× 只能称作“官方 benchmark 协议
复现”，不能称作最佳或完全公平的 FLA Triton 对照。公平对照在 inference mode
下显式设置 `FLA_FLASH_KDA=0, safe_gate=True`，由 `experiments/bench_c1.py` 单列。
同一 job、每格 1000 个样本的 paired 结果如下；比值为 FLA/Flash，>1 才表示
FlashKDA 更快：

| H | fixed：Flash / FLA / 比值 | mixed：Flash / FLA / 比值 | 8×1024：Flash / FLA / 比值 |
|---:|---:|---:|---:|
| 96 | 1.7705 / 3.0412 ms / 1.72× | 1.5255 / 3.1600 ms / 2.07× | 1.2286 / 3.0497 ms / 2.48× |
| 64 | 1.6194 / 2.1336 ms / 1.32× | 1.1444 / 2.2138 ms / 1.93× | 0.8314 / 2.0369 ms / 2.45× |
| 12 | 1.3630 / 0.8642 ms / **0.63×** | 0.5867 / 0.6107 ms / 1.04× | 0.2644 / 0.5078 ms / 1.92× |

绝对时间受不同 Slurm job 的卡频/节点负载影响，不能把这张表的 1.7705 ms 与旧 job
的 0.9999 ms 直接当代码回退；表内 paired 比值才用于公平 FLA 结论。它揭示一个
重要边界：TP8 对应的 H=12 fixed 下，低 grid 的 FlashKDA 反而慢于 safe-gate FLA；
序列拆分后才重新领先。

相同总 token 下，B300 上 fixed→8×1024 的加速分别为 1.36× 和 1.84×，方向与
GB200 一致，再次支持 K2 并行度不足的判断。

总 token 不变，K1 grid 量级也相近；明显变化是 K2 grid 从 `1×H` 增到 `8×H`，
而每个 CTA 内串行 chunk 数从 512 降到 64。H=64 固定长度增加到 H=96 时，工作量
增加 50% 而耗时只增加 9.89%，也说明 H=64 时没有提供足够并行度。TP8 的每卡 H=12
会更严峻；单卡 H12 实测已验证这个 kernel-grid 趋势，但真实 TP 通信与端到端场景
仍需另行验证。

## 2. 讨论点 1：为什么是 CHUNK=16

### 2.1 数值范围最先破

gate 激活范围是 `[-5, 0]`。源码同时构造 `exp(cumsum(g))` 与
`exp(-cumsum(g))`，所以 worst case 的指数绝对值是 `5C`：

| C | `exp(+5C)` fp32 | `exp(-5C)` 转 bf16 | 结论 |
|---:|---:|---:|---|
| 16 | 5.54e34 | 1.81e-35 | 有限，距 fp32/bf16 上界的 ln 余量约 8.7 |
| 32 | inf | 0 | 硬失败 |
| 64 | inf | 0 | 硬失败 |

因此从 16 到 32 时，**数值范围先于代数正确性和 tile 匹配失效**。即使典型 gate
没有触发 worst case，kernel API 允许该输入；大 chunk 必须像 Flash Linear
Attention 一样分段 rescale，不能只改模板常量。

### 2.2 Neumann 展开的代价

严格下三角 C×C 的 L 满足 `L^C=0`。C=16 时源码用 6 次 16³ GEMM 构造
`L²/L⁴/L⁸` 并逐次更新 INV。C=32 需再加 `L¹⁶` 的两次 GEMM，C=64 再加
`L³²`：

| C | inverse GEMM 数 | K1 FLOP/token | K2 FLOP/token | 合计 FLOP/token |
|---:|---:|---:|---:|---:|
| 16 | 6 | 11,264 | 106,496 | 117,760 |
| 32 | 8 | 32,768 | 114,688 | 147,456 |
| 64 | 10 | 114,688 | 131,072 | 245,760 |

这张表由 [`chunk_model.py`](experiments/chunk_model.py) 根据源码 GEMM 形状生成，
不含 pointwise/copy。C=32 的每 token GEMM 工作已多 25.2%，C=64 多 108.7%。

### 2.3 workspace 与并行度也没有免费收益

K1/K2 中间 workspace 每 tile 为
`3*C*D*2 + D*4 + 2*C*C*2` 字节。T=8192、H=96、D=128 时，C=16/32/64
的**有效 tile payload**分别为 648/684/774 MiB（不含 wrapper 保守多分配的 N 个
tile 和小 prefix buffer）；大 chunk 没有降低 workspace，反而因为 C² 的 INV/Mqk
增长而上升。K1 blocks 从 49,152 降至 24,576/12,288，K2 blocks 仍是 N×H。

## 3. 讨论点 2：tcgen05 与 CHUNK=16 是否匹配

NVIDIA 的 dense BF16 `tcgen05` 约束是：CTA group 1 的 M∈{64,128}、N 为
8--256 且步长 8、K=16。参见 [CUTLASS tcgen05 API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html)
和 [Blackwell GEMM 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)。

逐阶段匹配如下：

| 阶段 | 逻辑 GEMM | 不重构时的 tcgen05 情况 |
|---|---|---|
| K1 L/Mqk | M=16, N=16, K=128 | M=16，25% 行利用率 |
| K1 inverse | M=16, N=16, K=16 | M=16，25% 行利用率 |
| K2 k/q × state | 两个 M=16, N=128, K=128，共用 B | 可沿 M 合成 32，仍仅 50% |
| K2 INV/Mqk × U | M=16, N=128, K=16 | M=16，25% |
| K2 state delta | M=128, N=128, K=16 | M=128，形状自然匹配 |

最后一项约占 C=16 K2 GEMM FLOP 的 30.8%，是唯一适合做定点原型的阶段。但
tcgen05 是单线程发射、异步执行、累加器驻留 TMEM；读取必须经 `tcgen05.ld` 到
寄存器，不能由 TMA 直接搬走。这与 [assignment02.pdf](../../assignment02.pdf)
M3 的数据通路一致。当前 K2 则把 U 保持在寄存器并用 `MOVM_T` 在 Phase 3/4/6
间复用。换指令会改变中间态所有权和同步，不能按“相同 FLOP、峰值更高”估收益。

微基准 [`mma16_vs_tcgen05.cu`](experiments/mma16_vs_tcgen05.cu) 做指令粒度对照：
精确 m16n16k16 vs 补零到 m64n16k16，计入 TMEM alloc/commit/wait/load。
tcgen 路径只初始化指令实际读取的 K=16 数据，不再为仅作为 descriptor stride 的
`PAD_K=64` 无效列计时；M 的 48 个物理 padding 行仍显式置零。
它同时报告 physical 与 useful TFLOP/s。下表取最接近真实替换的一次 MMA/launch；
两条路径均已验证输出有限且 `max_diff=0`：

| 路径 | 时间 | useful TFLOP/s | useful speedup |
|---|---:|---:|---:|
| mma.sync m16n16k16 | 4.117 us | 0.002 | 1.00× |
| tcgen05 m64n16k16（75% padding） | 6.156 us | 0.001 | 0.669× |

即使把同一 MMA 在一次 launch 内重复 4096 次来摊薄启动开销，tcgen 的 useful
speedup 仍只有 0.889×（81.904 vs 92.130 us）；padding 的 physical throughput
不能转化为 useful 收益。

另一个 [`state_delta_tcgen05.cu`](experiments/state_delta_tcgen05.cu) 专测唯一自然
匹配的 m128n128k16 state-delta：对照路径用 4 warps 完成 64 个 m16n16 atom，
tcgen 路径用一条 m128n128 指令并计入 TMEM drain。这样不会因为只测最差的
padding case 而预设结论：

| state-delta 路径 | 时间 | TFLOP/s | speedup |
|---|---:|---:|---:|
| 4-warp mma.sync | 10.253 us | 0.051 | 1.00× |
| tcgen05 m128n128k16 | 22.532 us | 0.023 | 0.455× |

重复 256 次时 tcgen 可达到 2.358×，说明 instruction 本身有吞吐潜力；但这个
microbench 是单 CTA、单 launch，重复使用同一 A/B 和同一累加器，且只 drain 一次。
真实 K2 的 512 个 chunk 会更换输入、递推 state；TMEM alloc/dealloc 有机会移到
循环外，数据供给和必要 drain 又可能留在循环内。因此一次与 256 次分别是当前实现
的两个实验端点，都不能直接预测端到端加速。原始四档数据在
`results/job_13306/{mma16_vs_tcgen05,state_delta_tcgen05}.txt`。旧 job 12186 会为
指令不读取的 `PAD_K` stride 列填零，只保留为历史记录，不再作为主表。

## 4. 讨论点 3：chunk 间依赖下还能从哪里取并行度

先把 token 级 recurrence 写成 V-first state `S∈R^{V×K}`：

```text
S_decay = S_(t-1) diag(exp(g_t))
e_t     = v_t - S_decay k_t
S_t     = S_decay + beta_t e_t k_t^T
o_t     = S_t (scale q_t)
```

这也给出 V-split 的正确性依据：沿 V 行把 `S、v、e、o` 分成两个 64-row block 后，
每个 block 只读共享的 K 维 `q/k/g/beta`，而 `S_decay k_t` 与外积
`e_t k_t^T` 都逐 V 行独立；两个 CTA 不需要彼此的 state，也没有跨半幅归约。相反，
沿 K 拆分会在 `S k` 和输出上引入归约，不能用同样方式独立递推。

| 候选 | 正面论据 | 反例/风险 | 决策 |
|---|---|---|---|
| 多 head/CTA | 可尝试凑 M；减少重复控制 | 不同 head 的 B/state 不同，不能直接拼成一次 GEMM；state+pipeline 约 98 KiB/head，且 CTA 数更少 | 固定长/TP8 场景拒绝 |
| persistent K2 | 省启动和 state 往返 | 当前 K2 已是一 CTA/head 跨全部 chunk 的 persistent recurrence | 已经做了 |
| K1+K2 重新融合 | 消掉约 1.3 GiB workspace 写读 | 官方早期 fused prototype 因 K1 被 K2 低并行度拖住，端到端至少慢 15% | 不重复旧路线 |
| 每 head 两个独立 CTA、沿 V 拆分 | 每个 CTA 负责 64 个 V 列；两半 state/output/delta 完全独立，把 TP8 fixed grid 从 12 提到 24 | 重复加载 q/k/INV 等左操作数；需要半宽 output/state store | **挑战原型** |
| cluster producer/consumer | K1 producer 与 K2 consumer 用小环形缓冲，可能兼顾并行和少写 HBM | 跨 CTA 生命周期、反压和 varlen 调度复杂；集群规模固定 | 研究项，不进首版 |

V-split 原型不需要 chunk 间或 CTA 间同步：每个 CTA 只递推自己的 64 个 V 行；代价
来自重复的左操作数流量和新的半宽 store，而不是 cluster barrier。它的 kill criteria：
固定 T=8192、H∈{12,64,96} 至少 10% 加速，varlen
不得回退超过 3%，workspace/数值结果不变。达不到就停止。

原型由编译开关 `FLASH_KDA_K2_VSPLIT=1` 启用，默认值为 0，不改变上游路径。
启用时 K2 从 `grid=(N,H,1)` 改成 `(N,H,2)`，每 CTA 从 4 个 MMA warp 减为 2 个，
`blockIdx.z` 决定其 64 个 V 列；K1 和 workspace 格式不变。两个 CTA 仍各自加载完整
左操作数，output 与 bf16/fp32 final state 只回写各自的半幅，避免写冲突。核心改动
位于 `fwd_kernel2.cuh`、`fwd_launch.cu`、`fwd.h` 和 `setup.py`。

### 4.1 挑战结果与 kill decision

最终验证从同一 pin 干净编译 `k2_vsplit_enabled=False/True` 两条路径。两边的官方
H96 fixed/mixed-varlen 大形状均对 kernel-matched `torch_ref` 逐元素 exact-match；
此外每条路径各通过 20 项代表矩阵，覆盖 H=1/4/12/64/96、T=1/15/16/17/33 的
边界与尾块、B=2、混合 varlen，以及 bf16/fp32 的 in+out、in-only、out-only 和
no-state 模板。独立参考方面，题目给出的纯 PyTorch `naive.py` 与显式禁用
FlashKDA 自动派发、`safe_gate=True` 的 Triton `chunk.py`，各自 fixed/varlen 共
4 项均低于预注册的 1% rel-RMS 门槛（观测 output/state 为 0.44%--0.58%）。

因此“实现与 kernel 舍入完全一致”和“与独立 KDA 语义参考在阈值内一致”是两条
分开的证据链。完整日志是 `results/job_13306/default_*.txt` 与
`results/job_13306/vsplit_*.txt`。性能仍取 fp32-state 行：

| H / case | baseline | V-split | baseline / V-split |
|---|---:|---:|---:|
| 96 / fixed | 0.9999 ms | 2.7582 ms | 0.363× |
| 96 / mixed | 0.8721 ms | 3.0041 ms | 0.290× |
| 96 / 8×1024 | 0.7327 ms | 2.8235 ms | 0.260× |
| 64 / fixed | 0.9099 ms | 2.4087 ms | 0.378× |
| 64 / mixed | 0.6592 ms | 2.1994 ms | 0.300× |
| 64 / 8×1024 | 0.4947 ms | 1.8830 ms | 0.263× |

上表是旧 job 的官方协议结果。最终 job 13306 在同一作业中依次计时默认/V-split，
绝对时间整体变慢但 paired 比值复现同一结论，并补上 kill criteria 中最关键的 H=12：

| H / case | final default | V-split | default / V-split |
|---|---:|---:|---:|
| 96 / fixed | 1.7705 ms | 4.8048 ms | 0.368× |
| 96 / mixed | 1.5255 ms | 5.2250 ms | 0.292× |
| 96 / 8×1024 | 1.2286 ms | 4.9655 ms | 0.247× |
| 64 / fixed | 1.6194 ms | 4.2019 ms | 0.385× |
| 64 / mixed | 1.1444 ms | 3.8249 ms | 0.299× |
| 64 / 8×1024 | 0.8314 ms | 3.3176 ms | 0.251× |
| 12 / fixed | 1.3630 ms | 3.9566 ms | **0.345×** |
| 12 / mixed | 0.5867 ms | 1.7074 ms | **0.344×** |
| 12 / 8×1024 | 0.2644 ms | 0.8675 ms | **0.305×** |

它在所有实测形状都触发 kill criteria。即使没有 state store，H96 fixed 也从
1.0290 ms 退化到 2.6688 ms，说明重复输入流量与逐元素 output epilogue 已足以
淹没多一倍 CTA 的收益；bf16 state 的逐元素回写更慢到 5.6732 ms。这个结果否定
的是**当前 correctness-first 实现**，并不证明带半宽 TMA descriptor、32-lane
coalesced epilogue 和只加载半幅 V/state 的版本绝对不可能加速；但这些都已是新的
数据通路工程，不能把 V-split 描述成低成本优化。旧构建日志在
`results/job_12207/`；最终双路径构建、完整对拍和 paired 性能日志在
`results/job_13306/`。

## 5. 讨论点 4：compute-bound 还是 memory-bound

按 C=16 源码 GEMM 计数，T=8192/H=96 有 92.61 GFLOP。官方结果生成器选取
`fp32 state` 一行，因此算法级 global-memory traffic 模型包含 q/k/v/g、out、fp32
state 输入输出及 648 MiB workspace 的一次写+一次读，共 2.3782 GB；若这些都到
HBM，AI 约 **38.9 FLOP/B**。即使乐观假设 workspace 全被 L2 吸收，剩余强制
HBM 字节约 1.019 GB，AI 也只有 90.9 FLOP/B。
NVIDIA 给出的 HGX B300 八卡 BF16 是
36 PFLOP/s sparse（dense 为一半），即单卡 dense 2.25 PFLOP/s；单卡 HBM 带宽
最高 8 TB/s，见 [HGX 规格](https://www.nvidia.com/en-in/data-center/hgx/)
与 [NVIDIA HGX reference architecture](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)。
因此机器平衡点是 281.25 FLOP/B，本负载位于 memory side。

用官方 1.0087 ms 代入只是 sanity check：91.8 useful TFLOP/s（峰值约 4.1%）和
2.35 TB/s 有效字节率（峰值约 29.4%）。因此更准确的描述是：**不是
Tensor-Core-compute-bound；K2 更可能是依赖链/并行度不足造成的 latency-bound，
并伴随大量数据搬运，而非已经打满 HBM。**

B300 ncu 必须按 K1/K2 分开报告：

- `gpu__time_duration.sum`：两 kernel 时间占比；
- DRAM read/write bytes 与 `dram__throughput...pct_of_peak`：验证字节模型及是否打满 HBM；
- `lts__t_bytes.sum`/L2 hit rate：workspace 是否被 L2 吸收；
- tensor pipe active、SM throughput：排除 Tensor Core 饱和；
- active warps、achieved occupancy、SMEM/register occupancy limit：区分资源限制与 grid 不足；
- waves per SM / grid size：解释 fixed 与 varlen 差异。

本组 `ncu --set full` 的核心结果如下（时间是单次 kernel 的 replay 校正结果）：

| case/kernel | 时间 | DRAM peak | SM peak | tensor pipe | active warps | waves/SM |
|---|---:|---:|---:|---:|---:|---:|
| fixed K1 | 273.856 us | 58.72% | 70.07% | 5.62% | 96.61% | 41.51 |
| fixed K2 | 719.008 us | 19.50% | 21.84% | 20.14% | 9.37% | 0.32 |
| 8×1024 K1 | 280.480 us | 57.32% | 69.50% | 5.46% | 96.43% | 42.16 |
| 8×1024 K2 | 416.608 us | 36.29% | 40.30% | 34.77% | 16.84% | 2.59 |

fixed 的 K2 占两 kernel 时间 72.4%，但 DRAM、SM、tensor pipe 和 active warps 都
很低；它不是打满某个峰值单元，而是 96 个长依赖 CTA 只有 0.32 wave/SM。拆成
8 个 sequence 后，主导的逐 token GEMM 数不变，但更多 sequence state I/O 使总
字节并非严格不变；waves/SM 提到 2.59、时间下降 42.1%，各资源利用率反而提高。
这是“并行度/依赖链优先于换 Tensor Core”的直接硬件证据。原始
report/CSV 在 `results/job_12186/ncu_{fixed,varlen_8x1024}.*`。

metric 名称随 ncu 版本变化，先用 `ncu --query-metrics` 查本机名称；完整 `--set full`
report 已由脚本保留，避免因为某个别名变化丢数据。M4.5 的
`in_proj_qkvgfab (N=6288,K=7168)` 是前置投影，不是上述 recurrent kernel；它的
瘦 GEMM数据用于说明“小并行维下峰值 Tensor Core 不等于端到端收益”，不能直接
替代 C1 profile。

## 6. 讨论点 5：bf16 state 精度如何验证

官方 exact-match `torch_ref` 复刻了同一套 bf16 中间舍入，只能验证实现一致性，
不能证明“bf16 state 相对 fp32/fp64 足够准”。验证要分三层：

1. **状态隔离器**：fp64 state 为 gold；同一 fp32 FMA 更新分别在 chunk 间保存
   fp32/bf16。扫 T、gate 衰减、更新幅度、抵消构造和 seed。
2. **算子端到端**：用自写纯 PyTorch fp64 recurrence、FLA chunk、FlashKDA 三方
   对拍；`fused_recurrent_kda` 的 Triton kernel 内部使用 fp32，不能称为 fp64 gold；
   覆盖 fixed/varlen、T=1K/8K/64K、随机和 gate/beta 饱和、非零 initial state。
3. **模型级**：真实层分布和长上下文上比较 logits KL/top-1、关键 benchmark、跨层
   状态范数漂移；阈值由 fp32 backend 的模型波动确定，而不是只拍一个 atol。

指标包括按 token window 的 relative RMS、max abs、cosine、final-state error、
NaN/Inf，以及误差随 token/层数的斜率。

本地运行的 [`state_precision.py`](experiments/state_precision.py) 是第 1 层的可复现
合成隔离实验（D=128、C=16、rank-16 update）。不同 T/gate 现在复用相同随机流，
每格报告 3 个 seed 的均值±样本标准差：

| T | gate/token | bf16 final rel-RMS | bf16 probe rel-RMS |
|---:|---:|---:|---:|
| 1024 | -1e-4 | 0.931±0.008% | 0.940±0.021% |
| 2048 | -1e-4 | 1.278±0.010% | 1.272±0.024% |
| 8192 | -1e-4 | 2.298±0.025% | 2.309±0.037% |
| 8192 | -2e-2 | 0.243±0.004% | 0.252±0.011% |
| 8192 | -5e-1 | 0.166±0.001% | 0.171±0.014% |

fp32 control 的 final rel-RMS 均值为 2.5e-8--5.5e-7。结论是 bf16 误差受有效记忆
长度强烈控制；常规衰减下小，弱衰减长记忆时持续增长。因此这组数据支持“需要真实
gate 分布验证”，不单独支持“无可测精度损失”的模型级结论；它也不包含真实 KDA
中 state-dependent delta 的全部数值相关性。

## 7. 讨论点 6：如果我们是作者，v2 怎么发

### 支持 SM100 专版的论据

- state-delta 的 m128n128k16 天然匹配 tcgen05，且官方文档给出的 bf16 吞吐高于
  前代路径；
- 两个独立 CTA/head 的 V-split 有机会修复 fixed/TP 下的 grid 不足；
- B300 是 K3 的重要部署目标，10% 以上的单算子收益乘 69 个 KDA 层值得维护分支。

### 反对的论据

- 大多数阶段 M=16，直接 tcgen 只有 25% 行利用率；
- TMEM 数据通路破坏已有寄存器级融合，端到端收益不能由峰值外推；
- CHUNK 扩大先撞数值范围，再增加 inverse 复杂度和 FLOP/token；
- 当前一个 SM80 MMA 主线覆盖 SM90/100/103/120，而专版增加编译体积、测试矩阵、
  数值漂移和 CUTLASS/PTX 维护成本；
- 官方 fixed/varlen 差异指出更先要解决的是 K2 并行度。

### 发布决策

基于当前证据，v2 不发布全量 sm100a replacement，也不 dispatch 当前 V-split。保留 portable
默认路径；若后续继续研究，只接受两个独立分支：一是先实现半宽 TMA/coalesced
epilogue 再重测 V-split，二是把 state-delta 的 tcgen05 TMEM 生命周期跨多个工作
单元摊薄。任何分支仍须通过正确性矩阵和 10%/3% kill criteria；在此之前所有形状
继续使用 SM80 MMA，避免为了架构标签牺牲实际性能。

## 8. 答辩 10 分钟建议

1. 1 分钟：结论与决策门槛。
2. 2 分钟：源码数据流 K1→workspace→K2，指出 SM80 atom 和串行 recurrence。
3. 2 分钟：CHUNK 表——C=32 数值先爆，FLOP/workspace 同时上升。
4. 2 分钟：tcgen tile 表 + microbench 数据，强调 physical/useful 的区别。
5. 2 分钟：ncu 的 K1/K2 roofline、fixed/varlen 对照、2-CTA 候选。
6. 1 分钟：bf16 state stress case 与最终 v2 决策。

最可能被问的反例是“为什么不把四个 head 打包到 M=64”。回答：不同 head 的 B/state
不同，普通 GEMM 沿 M 打包要求共享 B；做 block diagonal 会引入更多 padding，且每
head 约 98 KiB shared-memory 工作集和更少 CTA 会进一步恶化 fixed/TP 并行度。
