# DS4 vLLM 版本审计（零-GPU）— TASK-1.1.12

日期 2026-07-12。回应 bayan 22:29 三问：①ray 臂验尸 ②为什么 cw 侧 pin vLLM 0.20.2 这么老 ③升级到最新可跑 DS4 的 vLLM 是否一箭双雕（顺带修 multiproc 可见性 / 泄漏）。全部零 GPU，纯审计。

---

## ① ray 臂验尸：翻车不是 vLLM，是 Ray 集群 bootstrap

- 8 卡 load-only 双臂：MP（默认 multiproc）= job **13875737 COMPLETED rc=0**；RAY（`distributed_executor_backend=ray`）= job **13875738 FAILED rc=1**（elapsed 6:52）。
- RAY 臂真实死因（stdout 尾部堆栈）：vLLM `ray_executor.py:78 _init_executor → initialize_ray_cluster → ray.init(address=...)` →
  `ray/_private/node.py:377` **`Exception: The current node timed out during startup. This could happen because some of the raylet failed to startup or the GCS has become overloaded.`**
- 结论：**ray 臂翻车 = Ray raylet/GCS 启动超时**，发生在 vLLM ray executor 试图 attach 一个 Ray 集群时；load-only harness 是单进程 `LLM(...)`，没有预先起好的 Ray head/cluster，`ray.init(address=...)` 等不到 GCS。**不是** vLLM 推理能力问题，**不是** DS4 模型问题。bayan「别急着怪 vLLM」判断正确。
- 附注：两个 overlay 混用可见于堆栈——`ray` 来自 `mlite-2604-verl-dsa-sm90-overlay`，`vllm` 来自 `mlite-2604-ds4-vllm020-thin`。ray executor 路径要真用，须在 harness 里显式起 Ray cluster（`ray start --head` 或 placement group），并对齐两 overlay 的 ray 版本；这是 harness/config 工作，不是执行器回退。

## ② 为什么 cw 侧 pin vLLM 0.20.2（1.1.13 选型依据）

来源：TASK-1.1.13（2026-06-16，claude），dead_ends/mlite-env-setup。

- cw 容器基座 = **NGC 26.04 / torch 2.12.0a0 / CUDA 13.2 / SM90(Hopper)**。
- DS4(deepseek_v4) 是全新架构，**vLLM ≥ 0.20.0 才有 `DeepseekV4ForCausalLM`**（vLLM PR#40760，blog 2026-04-24）；容器原装 vLLM 0.12.0 registry 无 DS4。
- 所有 pypi vLLM 0.20–0.23 wheel 都 **pin torch==2.11.0**，没有任何版本 build against 容器的 torch 2.12/CUDA13.2。DSA 训练栈（flash_mla/cutlass/cudnn 全 build on torch2.12）不能降 torch。
- 0.20.2 之所以被选中：它的 `_C.abi3.so` 相对 torch2.12 **只缺 1 个 ABI 符号**（`at::cuda::getCurrentCUDABlasHandle()` void 版），可用一个 env 层 LD_PRELOAD shim（`libvllm_torch212_abi_shim.so`）补上——即「能最省力 ABI-兼容到容器 torch2.12 的**最低** DS4-支持版本」，**不是**「最新」。
- 已 GPU 证过：job **12864592**（8×H100 TP8 COMPLETED rc=0）DS4_VLLM_ROLLOUT_SMOKE_PASSED，真加载 DS4-Flash fp8 159GB + 生成连贯文本。
- **qwen3.5 对比（bayan「q35 用 vllm023 镜像」）**：q35 farm 用不同镜像、且 q35 **不需要 DS4 支持**，两条栈不可复用/不可比——DS4 overlay 是为 DS4 专门现搭的。

## ②补 — GB200 track（1.1.22.1）其实已经做了「最新 vLLM」DS4 overlay，为什么 cw 没对齐

来源：TASK-1.1.22.1（2026-07-08，codex），已过 review。

- GB200 栈：**NGC 26.06 / torch 2.13.0a0 / CUDA 13.3 / SM100(Blackwell)**；vLLM **`cd0de48d0883...`（main 分支，≈ post-0.24，源码构建，无 wheel、无 ABI shim）**；DS4-Flash rev `60d8d707`。
- 已 GPU 证过：job **4281662**（4×GB200 TP4 COMPLETED rc=0）DS4_GENERATE_SMOKE_OK 生成 16 token。SM100 后端 = UE8M0 DeepGEMM + sparse_attn_indexer + TileLang mHC（MegaMoE 未选中→回退 FlashInfer TRT-LLM fused MoE）。
- 可复现构建脚本在本仓分支：`scripts/gb200-vllm-ds4/`（versions.env 锁 NGC26.06/vLLM cd0de48/model 60d8d707 + import/download/build_overlay/smoke）。
- **为什么 cw 没对齐**：GB200 overlay 是 **SM100/Blackwell + NGC26.06/torch2.13 源码构建**，跑在另一个集群（oci-hsg）。cw 是 **SM90/Hopper + NGC26.04/torch2.12 + wheel+ABI-shim**。把新 vLLM 落到 cw 需要**对 H100 重新源码构建**（换 NGC26.06 容器或 source-build against 容器 torch2.12），2026-06-16 建 cw overlay 时 GB200 的新栈还没出（那是 07-08）。纯属两条轨时间差 + 架构差，没对齐 = 待办，不是矛盾。

## ③ 升级最新 vLLM 是否一箭双雕（修 multiproc 可见性/泄漏）

上游事实（web 核实）：
- 最新稳定 = **vLLM v0.21.0（2026-05-15）**，在 0.20.0 DS4 基座上做 DS4 稳定化：含 **DS4 OOM fix #44914**、MTP projection、KV-cache dtype 修正、Blackwell TOKENSPEED_MLA 后端。GB200 已用的 cd0de48 更新（post-0.24）。
- **#44914 的实际内容**：DS4 MoE **专家权重创建期（init）** 的小额 OOM（要 1008 MiB 但只剩 979 MiB），修法=给 DS4 MoE 层套上 #41184 的量化处理类。**这是 init 期 expert 创建 OOM，不是我们的 resync all_gather 兄弟驻留 OOM**——相关性弱。
- Ray CVD-overwrite bug **#25113**（Ray 把每个 TP worker pin 单卡后，vLLM 又把 `CUDA_VISIBLE_DEVICES` 覆盖成整节点全卡）确实存在，**正是**旧「可见性泄漏」假设的上游对应物；但它在 **ray executor** 路径。

★关键校正（别让升级期望建在被证伪的假设上）★：
- 我们自己的 **8 卡 load-only A/B（jobs 13874307 baseline / 13874308 NCCL_P2P_DISABLE）已决定性 REFUTE「vLLM multiproc 可见性泄漏」是兄弟驻留根因**：两臂 GPU0 均**零兄弟**，每卡恰 1 进程。7×2.51GiB 兄弟是**真实 colocated resync 权重 all_gather**（update_weights 跨 TP8 gather 真权重）触发的，load-only（dummy 权重）结构上照不到。
- 因此：**升级 vLLM 不是 128 卡 resync-OOM 的已证银弹**。上游 changelog 里我能找到的 DS4 修复（#44914 init-OOM、#25113 ray-CVD）都不直接命中「resync all_gather 峰值 + colocated 兄弟」这个已坐实机制。
- 但升级仍**独立值得**，理由有三：(a) 0.20.2 是 wheel+ABI-shim 的脆弱最低版，0.21.0+ 有 DS4 稳定化与 OOM 修复；(b) ray executor 路径（bayan 定的 C 臂唯一路线）在新版 + 正确 Ray cluster 起法下更可能通；(c) 与已验证的 GB200 栈（cd0de48）对齐，去掉 ABI shim。

## 建议（供 bayan 裁，均需 human 定版本/预算——属新构建工作，非本 128 跑节点内活）

- **目标版本**：对齐 GB200 已 GPU 验证的 vLLM `cd0de48`（或至少 v0.21.0）。复用 `scripts/gb200-vllm-ds4/` 配方，改造到 cw H100。
- **构建路径二选一**（human 裁）：
  - **[A·推荐] cw 换 NGC 26.06 容器**（torch2.13/CUDA13.3），source-build vLLM cd0de48 for **SM90**，与 GB200 recipe 最大对齐、丢掉 ABI shim。代价=DSA 训练栈需在 NGC26.06 重验（flash_mla/cutlass/cudnn 重 build against torch2.13）。
  - **[B·小步] 保持 NGC 26.04/torch2.12**，只把 thin overlay 从 0.20.2 wheel 升到 0.21.0 wheel（仍需 ABI shim，符号集可能要重导）。拿到 #44914/DS4 稳定化，但仍是 wheel+shim 脆弱路。
- **诚实边界**：无论 A/B，**都不保证消掉 128 卡 resync 兄弟 OOM**（该机制我们已坐实为真实 all_gather 峰值，不是可见性 bug）。升级主要收益=解 ray 臂 + 去脆弱 + 对齐 GB200；resync-OOM 仍需 export 流驻留控制/兄弟机制的正交修法。
- 选版后：**8 卡 smoke（load+前向+ray 臂）→ 128 冒烟**（沿 SMOKE_EXIT_AFTER）。
