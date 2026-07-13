# 调研：vLLM 能否跑 DS4 TP16 + rollout 显存下降路线（零 GPU）

任务 TASK-1.1.22.7（bayan 出题）· 作者 claude_opus · 2026-07-13 · 零 GPU

> 一句话结论：**不能。** DS4（DeepSeek-V4-Flash）rollout 的 `ROLLOUT_TP` 天花板 = **8**，
> TP16 结构性不可跑，且**与硬件无关**（H100/GB200 同）。原因不是 DSA 的 `o_groups`，而是更底层的
> **FP8 128×128 block 量化**：TP16 把权重某维切到 128-block 边界以下，vLLM 的 deep_gemm
> post-load 除零/报错。上游 vLLM 明确把它当 **fundamental 约束**（issue #17569 **closed as
> not planned**），**没有** GQA-kv-head 式的 TP>约束复制方案；唯一在飞方向是 checkpoint 侧
> padding-to-128（llm-compressor #2286），不在 rollout/MLite 范围。
> → rollout 省显存**不能靠 TP16**，改走：① 廉价旋钮（gpu_mem_util↓/KV-FP8/seqlen 上限）
> 立即用；② 结构级 = resync 导出的**有界流式 per-expert EP all_gather** + vLLM 侧
> **expert-parallel**（上游对 DeepSeek MoE 的官方省显存手段）；sleep/wake 是高杠杆但有
> wake_up 回收完整性坑。

---

## 参考源声明（上游新鲜度铁律，fetch 于 2026-07-13）

| 参考 | 版本/commit | 用途 |
|---|---|---|
| DS4 实证栈（cw H100 x86） | `verl.vllm023.sqsh` → vLLM `0.23.1.dev0+g0fc695fc6` | job 13891915/13885909/13888695 全此栈 |
| GB200 canonical rollout 栈 | vLLM `main@cd0de48`（post-0.24 源码，NGC26.06/torch2.13/SM100，K-0161） | 32 卡 DS4 rollout 已验证栈 |
| 上游 vLLM | `vllm-project/vllm#17569`（**closed as not planned**），`fp8.py:478` load 期 raise | block-quant TP 整除 = fundamental 约束的权威表态 |
| 上游 llm-compressor | `#2286` Support Fp8 Block Quant for shapes not divisible by 128（closed，checkpoint 侧 padding） | 唯一在飞的"放宽整除"方向，属量化侧非 runtime |
| 上游 vLLM 文档/PR | Expert Parallel Deployment docs；PR#14068 moe fp8 block quant tuning | DeepSeek MoE 省显存官方手段 = TP≤8 + EP |

**关键：约束版本无关。** 0.23.1.dev0 上表现为 deep_gemm `ZeroDivisionError`（block 数 `g=0`），
新版/main 上表现为更友好的 `ValueError: output_size ... not divisible by weight quantization
block_n = 128`——**报错方式变了，天花板不变**。故"选了过期 vLLM 才 TP16 失败"不成立：升级 vLLM
**不**解锁 TP16。

**本任务边界**：零 GPU 调研，不复跑 GPU。TP16 的**实证击杀**已由兄弟 TASK-1.1.12 的
`job 13891915`（2 节点 16 卡 load-only 探针，绕过自家守卫让 vLLM 自证）完成；本报告 =
综合该实证 + 补齐上游新鲜度 + 把结论扩到 GB200/128 卡的即用决策。

---

## AC#1 — TP16 拒绝的精确机制（读 vLLM 0.23.1 + main）

### 两道约束，一浅一深

**（浅、MLite 侧）DSA `o_groups=8` 整除守卫**：`rollout_tp` 必须整除 `o_groups=8`
→ `rollout_tp ∈ {1,2,4,8}`。这是 **MLite 自家的 preflight**
（`deepseek_v4_rollout_load_only.py:203` + `run_deepseek_v4_gsm8k_grpo.sh:73`），**不是 vLLM
约束**。实证坐实：job 13891915 用 `DS4_ALLOW_TP_OVERSPLIT=1` 绕过此守卫后，
**vLLM 接受了 tp=16 config 并打了引擎 banner，config 期零拒绝**——即 vLLM 本身不认 `o_groups`。

**（深、vLLM 侧、硬数学）FP8 128×128 block 量化整除**：这才是真天花板。
DS4 权重是 FP8 block-quant，`weight_block_size=[128,128]`——**每个 128×128 block 共享一个
scale**。TP 把权重按 rank 切分时，**每个分片必须含整数个 128-block**，否则 block 的 scale
无处安放。tp=8 时每分片仍 ≥1 整块（已证可载，job 13885909，`DS4_VLLM_LOAD_ONLY_PASSED
rollout_tp=8`）；tp=16 把某维切到 <128 → block 数向下取整 = **0**。

实证栈精确崩点（job 13891915，vLLM 0.23.1.dev0）：
```
vllm/model_executor/layers/quantization/utils/fp8_utils.py:1037
  deepgemm_post_process_fp8_weight_block:  r = wq.size(0) // g
ZeroDivisionError: integer division or modulo by zero        # g（128-block 数）= 0
调用链: gpu_worker.load_model → process_weights_after_loading → fp8.py:428
        → flashinfer.py:175 → deep_gemm.py:96 → fp8_utils.py:1037
```
每个 worker 载权重时都崩（不是 config 期，是 load 期）。上游 main 对同一情形（分片非零但
不整除 128，如 192/128）在 `fp8.py:478` 抛 `ValueError: The output_size ... not divisible by
weight quantization block_n = 128`——**同一硬数学，只是我们的维恰好切到 0 触发的是除零**。

### 硬数学还是实现选择？→ 硬数学

vLLM 维护者对 issue #17569（Qwen3-235B-FP8 8 卡 TP 撞同一约束）的裁决是
**"closed as not planned"**，明确视为 **fundamental quantization constraint 而非可修 bug**。
理由：block 量化的 scale 是 per-128-block 的，跨 rank 切碎一个 block 就没有一致的 scale 可用。
这不是 vLLM 偷懒，是 block-quant 的定义使然。

### 上游有无 TP>约束的复制方案（类比 GQA kv-head replication）或在飞 PR？

- **没有 runtime 复制方案。** GQA 能复制 kv-head 是因为 kv-head 是可整份复制的独立单元；
  FP8 block 的 scale 绑定在 128×128 几何上，**不能像 kv-head 那样"复制一份塞给多余 rank"**——
  复制半个 block 没有意义。vLLM 侧不打算加（#17569 closed-not-planned）。
- **唯一在飞方向 = checkpoint 侧 padding。** llm-compressor #2286
  "Support Fp8 Block Quant for shapes not divisible by 128" 提议在**量化产 checkpoint 时** pad
  到 128 的倍数。即便落地，也只是让**原本不整除的维**能量化/能更高 TP；对 DS4 而言，要靠它解锁
  TP16 得把每个分片维 pad 到 tp=16 下仍 ≥128 —— 等于重量化整个 checkpoint、放大权重、且属**离线
  量化管线**，**不在 rollout/MLite/vLLM-runtime 范围**，不是本线可用杠杆。
- **上游对"MoE 想更宽却撞 block 整除"的官方答案 = TP≤8 + `--enable-expert-parallel`**
  （见 AC#2 路线 5、AC#3）。这才是替代 TP16 的正解，且已是 DeepSeek-V3/V4 部署标准配方。

**AC#1 小结**：TP16 死于 FP8 128-block 硬数学，与硬件/vLLM 版本无关，上游判 not-planned，
无 kv-head 式复制，唯一放宽方向在量化侧且不在本范围。**`ROLLOUT_TP ≤ 8` 是硬天花板。**

---

## AC#2 — rollout 显存替代路线（逐项：机制 / 省多少 / 改动面 / 风险）

**先钉住问题的形状**（据 job 13888695 判决书，128 卡首 resync OOM 内存分解，GPU0 总 79.11 GiB，
**仅差 128 MiB**）：
- vLLM engine 进程 **39.62 GiB** + 7× 兄弟 vLLM worker 各 2.51（=17.6）→ vLLM 侧 ~57 GiB
- 训练进程 **21.77 GiB**
- 首 resync **导出峰 24–36.4 GiB/rank**（dense `ep_gathered`，`hf_weights.py:_ep_all_gather`）
- `opt_resident_on_entry=False grad_resident_on_entry=False`（offload 正常，**非**优化器/grad 常驻）

bayan 的 TP16 本意 = 把 vLLM 的 39.62 腰斩到 ~20，给导出峰腾地。TP16 死了，改用下列杠杆
（**大头永远是"导出峰 × colocated vLLM 驻留"在同一张 GPU0 相撞**）：

| # | 路线 | 机制 | 省多少 GiB | 改动面 | 风险 |
|---|---|---|---|---|---|
| 1 | **gpu_memory_utilization 0.60→~0.45** | vLLM 按 `util×(total−weights)` 预留 KV；调低把预留还回空闲池 | GPU0 释 **~12 GiB**（0.15×79），远盖 128 MiB 缺口 | 1 个值 | 冒烟 seqlen（max_model_len=384）KV 极小，几乎零风险；**真 DAPO seqlen 须复核 KV 压力**别 OOM 在生成侧 |
| 2 | **max_model_len / max_num_seqs 下调** | 直接界定 KV cache 上限（与 #1 互补的另一面） | 与 KV 占用成正比 | 2 个 flag | 上限必须 ≥ DAPO 真实生成长度，**不能截断推理轨迹**（截断=样本污染，见停止符合同族坑）；是正确性旋钮非白送 |
| 3 | **KV cache FP8（`kv_cache_dtype=fp8`）** | KV 存 FP8 → ~半 KV 显存（vLLM 0.23 已支持） | 半个 KV 占用（真 seqlen 才显著，冒烟可忽略） | 1 flag | **rollout logit 数值漂移** → 影响 DAPO on-policy 重要性采样比（rollout 须与 actor 打分一致）；须过 parity（接 [[TASK-1.1.22.2]] resync 数值一致性）。非 RL-免费 |
| 4 | **sleep/wake（vLLM sleep mode L1/L2）** | 训练/优化器窗口把 vLLM 睡下（L1 归还权重、L2 连 KV），rollout 前唤醒；导出峰期若 vLLM 不驻留，36 GiB 峰轻松装下 | 训练窗内最多释 **~57 GiB**（整个 vLLM 足迹） | verl/mlite colocated 编排须在 resync/train 相位睡/醒 vLLM，且理顺"唤醒 vs 权重下发"次序 | **wake_up 回收完整性坑（已知）**：export 缓冲释放只回 torch 缓存，vLLM cumem 分配器看不到 → 需 empty_cache-before-wake（见 [[mfsdp-resync-oom-emptycache-fix]]）。且 resync 是"往 vLLM 灌权重"，下发瞬间 vLLM 必醒，睡只能护住训练窗、护不住 handoff 峰——相位设计最精细。高杠杆但最易踩坑 |
| 5 | **vLLM 侧 Expert Parallel（TP8 + `--enable-expert-parallel`）** | DeepSeek MoE 上游标配：不 TP-切碎每个 expert 权重（高 TP 会切到 128-block 以下），而是把**整份 expert** 摊到各 rank；每 rank 少持 expert → MoE 权重显存降，且每份 expert 保持整 128-block（**不违整除**） | MoE 权重占 DS4 大头；EP over 8 → 每 rank ~1/8 experts。这正是 TP16 想达到却做不到的"降 per-GPU 权重" | vLLM flag；但 resync 导出合同须对齐 vLLM EP 布局（正好就是 `_ep_all_gather` 路径） | EP 改变推理 token 路由/all-to-all；colocated resync 须把 MLite expert 分片 → vLLM EP rank 正确映射。非平凡但**结构正确且上游背书** |
| 6 | **rollout PP（vLLM `pipeline_parallel_size>1`）** | 层切到多卡 → 每卡持更少层权重 | 权重/PP per GPU | vLLM pp flag + colocated 摆位（PP rank 落哪张物理卡 vs actor） | colocated PP 摆位复杂 + 流水气泡→rollout 延迟；DS4 resync 导出须处理 PP 布局。编排成本高于 EP，优先级低 |
| 7 | **结构级自研：resync 导出的有界流式 per-expert EP all_gather** | OOM 是**导出瞬时峰**（dense `ep_gathered` 36 GiB）非稳态；改 per-expert 流式（≤4 GB/块，vLLM 消费后即 drop）→ 削掉 36 GiB 尖峰。**训练侧（MLite），与 vLLM TP 无关** | 导出峰 36 → 每块 ~4 GiB | MLite `hf_weights.py:_ep_all_gather`/`_gather_expert` + `_stream_resync_export`（per-tensor 流式已在，残留是 dense 前置 gather） | 分块 gather 正确性；已有 TASK-1.1.19 bounded-streaming-exporter 打法可照。**这是消除悬崖的真结构修法** |

**注（已证伪杠杆，别再试）**：free-grad / optimizer-evict 对本配置 **empirically NULL**——
resync 入口 `opt/grad_resident=False`，VERL 已预先 offload，grad 根本不驻留 GPU（job 13840020/
13888695 双证；`grad_offload` flag 在 mlite engine 是死代码）。7×2.51 GiB 兄弟由**真 resync
all_gather** 触发（"vLLM 可见性泄漏"根因已被 8 卡 A/B **REFUTED**，job 13874307/8）。

---

## AC#3 — 结论：128 卡 colocated 推荐组合（供 bayan 拷打）

### 🟢 即用（零上游、低风险、今天就能配）
1. **`ROLLOUT_TP=8`**（硬天花板，已证 job 13885909）——**放弃 TP16，别再发 TP16 版**。
2. **`gpu_memory_utilization` 0.60→~0.45**：GPU0 释 ~12 GiB，冒烟缺口（128 MiB）绰绰有余。
3. **按真实 DAPO 生成长度收 `max_model_len`/`max_num_seqs`**（别截断推理轨迹）。
4. **真 seqlen 下开 `kv_cache_dtype=fp8`**，但**先过 rollout↔actor logprob parity**
   （挂 [[TASK-1.1.22.2]]）——数值不过就退回 bf16 KV + 靠 #2/#3 腾地。

> 冒烟/资格：仅 ①②③ 就能越过 job 13888695 的 128 MiB 悬崖。bayan 02:33 已定调本轮 = hack-CP
> 冒烟/资格，**正式数字须等 TASK-1.2.12.2 fused-CP 实装后 128 卡重跑**。

### 🟡 结构级自研（排期，去悬崖化）
5. **resync 导出改有界流式 per-expert EP all_gather**（AC#2 路线 7，MLite `hf_weights.py`）：
   把 36 GiB dense 峰削成 ~4 GB/块 —— **不管 vLLM 足迹多大都不再撞悬崖**，是唯一的根治。
6. **评估 TP8 + `--enable-expert-parallel`**（AC#2 路线 5）：这是 TP16 想做（降 per-GPU 权重）
   却做不到的**上游正解**；代价 = resync 导出合同对齐 vLLM EP 布局。
7. **（高杠杆、精细）sleep/wake L2**：训练窗释 ~57 GiB vLLM 足迹；**前置条件 = 修 wake_up
   回收完整性**（empty_cache-before-wake，[[mfsdp-resync-oom-emptycache-fix]]），且理顺 handoff
   相位。收益最大但坑最深，建议在 5/6 之后再上。

### 🔴 需上游 / 不推荐 / 已死
- **TP16 = 死**：FP8 128-block 硬数学，硬件无关（GB200 同），vLLM not-planned，无 kv-head 式复制。
- **checkpoint 侧 padding-to-128**（llm-compressor #2286）：唯一能"抬 TP 天花板"的方向，但属**离线
  重量化管线**、放大权重、不在 rollout/MLite 范围 —— **不建议**，用 EP（路线 6）替代其目的。
- **rollout PP**（路线 6-of-AC#2）：可行但编排成本 > EP，非首选。

### 一句话给 bayan
TP16 这条路走不通（硬数学，非版本/非硬件问题）；rollout 省显存的**即用组合 = TP8 +
util0.45 + seqlen 上限（+ 数值过关的 KV-FP8）**，**根治 = resync 导出有界流式 EP all_gather
（自研）+ TP8+EP（上游）**；sleep/wake 是压箱底大杠杆但要先补 wake_up 回收坑。

---

## 证据索引（全部可复查）
- `job 13891915`（TASK-1.1.12）2 节点 16 卡 load-only TP16 探针 → deep_gemm 除零 → TP16 死。
  判决书 `<worktree TASK-1.1.12>/.vicky/evidence/13891915-tp16-oversplit-verdict.md`
- `job 13885909`（TASK-1.1.12）8 卡 load-only TP8 → `DS4_VLLM_LOAD_ONLY_PASSED`（TP8 可载）
- `job 13888695`（TASK-1.1.12）128 卡真 RL → 首 resync EP all_gather GPU0 OOM 差 128 MiB。
  判决书 `<worktree TASK-1.1.12>/.vicky/evidence/13888695-resync-oom-verdict.md`
- 上游：vllm-project/vllm **#17569**（closed-not-planned）、llm-compressor **#2286**、
  vLLM Expert Parallel Deployment docs、PR #14068
- 记忆：`ds4-128card-resync-oom`、`ds4-native-vllm023-cw-3walls`、
  `mfsdp-resync-oom-emptycache-fix`、K-0161（GB200 canonical 栈）
