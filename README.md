CUDA C++ GEMM 优化实践：V1–V9 Benchmark 与 Nsight Compute 分析

## 1. 项目目标与实现范围

本项目以 **FP32、Row-major 的矩阵乘法** `C = A × B` 为对象，使用 CUDA C++ 逐步实现和比较多种 **SIMT / CUDA Core GEMM** 优化方案，并通过 CUDA Events Benchmark 与 NVIDIA Nsight Compute（NCU）分析性能。核心关注点是线程/寄存器分块、访存合并、Shared Memory 数据复用、双缓冲、`cp.async` 与 Bank Conflict。

- **GPU：** 两批均为 NVIDIA GeForce RTX 3090。
- **第一批（V1–V7）：**RTX 3090 ；开发环境为 CUDA 12.8。
- **第二批（V6–V9）：** RTX 3090 / sm_86；开发环境为 CUDA 12.8；目标架构 `sm_86`。
- **测试形状：** `M = N = K ∈ {256, 512, 1024, 2048}`；属于对齐、整除条件满足的方阵，不代表所有 Kernel 都支持任意非规则尺寸。
- **Benchmark：** 使用 CUDA Events 计时（既有测试程序采用预热 15 次、重复 100 次的方式）；比较自定义 Kernel 与 cuBLAS SGEMM 的设备端执行时间，不包含内存分配与主机/设备拷贝。不同批次的 cuBLAS 数值路径、GPU 状态等可能不同。
- **正确性：** 两批上传的全部 44 条矩阵测试记录均显示 `bad=0`。

## 2. 各版本简要介绍

| 版本 | 主要实现 / 优化点 | 说明 |
|---|---|---|
| **V1** | 1D Thread Tiling | 每线程沿输出矩阵一个维度累计多个元素，利用寄存器提高数据复用。 |
| **V2** | 2D Thread Tiling | 每线程计算二维输出子块，增加寄存器级复用；并非保证所有尺寸都比 V1 快。 |
| **V3** | `float4` 向量化访存 | 在 2D Tiling 基础上尝试 128-bit 向量化加载/存储，降低部分访存指令开销。 |
| **V4** | Warp Tiling | 显式组织 Warp 内线程的计算分工与输出 Tile，优化数据复用和线程映射。 |
| **V5** | Register Double Buffering | 采用寄存器级 Ping-Pong / 预取组织计算，尝试重叠加载与计算。 |
| **V6** | Shared Memory Double Buffering | 采用两个 Shared Memory Stage 的传统加载路径，进一步尝试隐藏 Tile 加载延迟；作为 V9 的基线。 |
| **V7** | `cp.async` 异步 Global→Shared 拷贝 | Ampere 架构上使用异步拷贝与双缓冲，作为 V8 的基线。 |
| **V8** | V7 + Shared Memory Padding | 调整 `cp.async` 路线的 Shared Memory A Tile 布局，减少 NCU 统计的 Shared Load Bank Conflict。 |
| **V9** | V6 + Shared Memory Padding | 调整传统双缓冲路线的 Shared Memory A Tile 布局，消除 NCU 统计的 Shared Store Bank Conflict。 |

## 3. Benchmark：两批结果分别比较

两批虽然都是 RTX 3090，但相同的 V6/V7 在两批的耗时差别较大。例如 2048³ 的 V6：第一批 **1.4690 ms**，第二批 **1.0228 ms**；V7：第一批 **1.5184 ms**，第二批 **1.1132 ms**。

### 3.1 第一批：V1–V7

![image-20261027202012087](.figures\image-20261027202012087.png)

单位：**ms**；表中为自定义 Kernel 的 CUDA Events 平均延迟，数值越低越好。

| 版本 | 256³ | 512³ | 1024³ | 2048³ |
|:---:|---:|---:|---:|---:|
| V1 | 0.0333 | 0.0646 | 0.4079 | 2.6936 |
| V2 | 0.0800 | 0.1572 | 0.4464 | 2.5400 |
| V3 | 0.0664 | 0.1300 | 0.3468 | 1.6512 |
| V4 | 0.0385 | 0.0755 | 0.2408 | 1.4873 |
| V5 | 0.0385 | 0.0753 | 0.2415 | 1.5268 |
| V6 | 0.0365 | 0.0715 | 0.2314 | 1.4690 |
| V7 | 0.0259 | 0.0490 | 0.2298 | 1.5184 |

**观察：** 2048³ 下 V1 为 2.6936 ms，V6 为 1.4690 ms（相对 V1 约 **1.83×**）；V4 为 1.4873 ms，V6 只略快于 V4。小尺寸并不呈现严格单调加速：例如 256³ 下 V2 比 V1 慢。说明更复杂的分块与数据复用会引入额外资源/指令开销，其收益依赖问题规模。

### 3.2 第二批：V6–V9

![V6-V9 Benchmark](.figures\image-20261027202054402.png)

单位：**ms**；同一批测试内可比较 Padding 前后性能。

| 版本 | 256³ | 512³ | 1024³ | 2048³ |
|:---:|---:|---:|---:|---:|
| V6 | 0.0281 | 0.0548 | 0.1882 | 1.0228 |
| V7 | 0.0197 | 0.0372 | 0.1754 | 1.1132 |
| V8 | 0.0171 | 0.0297 | 0.1513 | 0.9976 |
| V9 | 0.0284 | 0.0550 | 0.1881 | 1.1490 |

**观察：** 在 2048³ 下，V8 是本批四个版本中延迟最低的（0.9976 ms），V6 次之（1.0228 ms）。**V8 相比 V7 加速约 1.116×**；**V9 相比 V6 延迟增加约 12.3%**。同一种 Padding 思路在两条实现路径上得到相反的最终性能结果。

## 4. NCU 性能分析：V7 → V8（`cp.async` 路线）

### 4.1 问题与修改

V7 使用 `cp.async` 将数据从 Global Memory 搬运到 Shared Memory，并进行双缓冲。在此前的 NCU 分析中，V7 存在大量 **Shared Load Bank Conflict**。V8 为 A Tile 增加 Padding，改变共享内存地址映射，尝试减少冲突。

v7 Shared Load Bank Conflict:

![image-20261027200717278](.figures\image-20261027200717278.png)

V7 Excessive Shared Wavefronts:

![image-20261027200748678](.figures\image-20261027200748678.png)

v8 Shared Load Bank Conflict:

![image-20261027200811655](.figures\image-20261027200811655.png)

v8 Excessive Shared Wavefronts:

![image-20261027200825430](.figures\image-20261027200825430.png)

### 4.2 结果对比（2048³）

| 指标 | V7 | V8 | 变化 |
|---|---:|---:|---|
| Benchmark 延迟 | 1.1132 ms | 0.9976 ms | **降低 10.4%** |
| Registers / Thread（编译日志） | 103 | 105 | +2 |
| Static Shared Memory / Block（编译日志） | 8,192 B | 10,240 B | +2,048 B |
| NCU Shared Load Bank Conflicts | 33,554,432 | 0 | 对应计数归零 |
| NCU Source Counters：Excessive Shared Wavefronts | 约 12,582,912 | 约 16,777,216 | **没有消失，反而增加** |

### 4.3 原因分析

- **实测成立：** V8 的 Shared Load Bank Conflict 计数为 0，且在本批 2048³ Benchmark 中更快。
- **推断：** V8 的 Source Counters 仍然把大量 Excessive Shared Wavefronts 关联到 `cp.async` 对应的 `LDGSTS.E.128` 指令。它与普通 Shared Load/Store 的 Bank Conflict 计数不是同一口径，**不能说 V8 消除了所有 Shared Memory 事务开销**。
- **可能机制：** Padding 改善了部分 Shared Memory 访问的 Bank 映射，使相关访问更高效；但同时增加 Shared Memory 和少量寄存器开销。仅凭现有指标不能定量拆分每种因素贡献。

**结论：** 对 `cp.async` 路线，Padding 在此次测试中带来了净收益；但不同 NCU 指标要按指令类型与统计口径分别解释。

## 5. NCU 性能分析：V6 → V9（传统双缓冲路线）

### 5.1 问题与修改

V6 的 A Tile Shared Memory 存储采用转置式布局，原始数组类似 `As[2][BK][BM]`；V9 改为带 Padding 的 `As[2][BK][BM+4]`。目的是改变 Warp 内线程写入 Shared Memory 时的 Bank 映射，减少 Store Bank Conflict。

v6 Shared Load Bank Conflict:

![image-20261027201045815](.figures\image-20261027201045815.png)

v6 Occupancy:

![image-20261027201112872](.figures\image-20261027201112872.png)

v9 Shared Load Bank Conflict:

![image-20261027201124456](.figures\image-20261027201124456.png)

v9 Occupancy:

![image-20261027201134999](.figures\image-20261027201134999.png)

### 5.2 结果对比（2048³）

| 指标 | V6 | V9 | 变化 |
|---|---:|---:|---|
| Benchmark 延迟 | 1.0228 ms | 1.1490 ms | **增加 12.3%** |
| Registers / Thread | 71 | 80 | +9 |
| Static Shared Memory / Block | 8,192 B | 8,448 B | +256 B |
| Shared Store Bank Conflicts | 4,194,304 | 0 | 对应计数归零 |
| Theoretical Occupancy | 58.33% | 50.00% | −8.33 个百分点 |
| Achieved Occupancy | 49.58% | 43.69% | −5.89 个百分点 |
| Resident Blocks / SM（理论） | 7 | 6 | −1 |
| Active Warps / SM | 23.80 | 20.97 | −2.83 |
| Eligible Warps / Scheduler | 2.40 | 2.09 | −0.31 |
| No Eligible（Scheduler） | 31.63% | 35.67% | +4.04 个百分点 |

### 5.3 原因分析？

**确定的观察：** Padding 消除了 NCU 统计的 Shared Store Bank Conflict，但 V9 的寄存器和 Shared Memory 占用更高；理论/实际 Occupancy、活跃 Warp 和 Eligible Warp 数下降，最终 Benchmark 更慢。

**合理但尚未证明的解释：**

1. **资源约束与驻留能力：** 每线程寄存器从 71 增至 80、每 Block Shared Memory 增加，使 NCU 给出的驻留 Block 数从 7 降为 6。更少的驻留 Warp 可能削弱延迟隐藏能力。
2. **Warp 调度：** Eligible Warps / Scheduler 下降、No Eligible 比例上升，说明调度器更经常缺少可立即发射的 Warp；这是性能下降的相关现象，不能单独认定为根本原因。
3. **指令级代价：** Padding 改变索引计算、地址布局与编译器代码生成，可能带来额外整数指令、不同的 `LDS/STS` 指令形态或依赖链。**当前未完成逐指令 SASS 对比，不能把这些可能性写成已证实事实。**

**结论：** 优化某个 NCU 计数器并不保证端到端 Kernel 变快。V9 是一个有价值的负向实验：消除 Bank Conflict 的收益可能被资源占用和其他执行开销抵消，最终应以同批次 Benchmark 验证。

## 6. 从源码到 NCU 的分析方法

本项目形成了如下可复用流程：

1. **Benchmark 识别候选实现：** 先对 V1–V7 在多尺寸下测时延，关注较有潜力的 V6/V7 路线。
2. **NCU 定位问题：** 通过 Shared Memory 指标、Source Counters 与源码关联，定位潜在 Bank Conflict。
3. **提出可检验的修改：** 分别针对 V7（V8）和 V6（V9）增加 Padding，保留其他路线用于对照。
4. **再次 Benchmark + NCU：** 同时比较延迟、冲突指标、寄存器、Shared Memory、Occupancy 和调度指标。
5. **谨慎归因：** 区分已观察的计数器变化和推测的底层机制。若要进一步确认 Bank 映射或指令变化，可结合线程地址计算、PTX/SASS 和指令级统计。

## 7. 实验限制与可复现性

- **精度模式：** 自定义 GEMM 是 FP32 SIMT 路线。
- **输入规模有限：** 当前 Benchmark 只覆盖四种对齐方阵；代码中部分 `float4` 和边界处理假设可能不适用于任意 M/N/K。
- **NCU 数据来源：** 本次上传的ZIP包包含 Benchmark 和第二批编译日志。NCU数据来源是ncu的原报告，并又NVIDIA Nsight Compute的GUI界面解析。
- **计时方式：** CUDA Events 重复调用平均耗时并非严格统计置信区间；没有保存多次独立运行的方差、GPU 锁频和功耗记录。

**项目核心认识：** GEMM 优化是一项多因素权衡：降低 Bank Conflict、提高数据复用、减少访存和增加并行度都可能带来收益，但资源占用、指令执行和调度效率也可能改变。
