# GDN CP 上游对照:本地 `gdn_cp_mode=sharded` vs 上游实现

> TASK-1.1.7.1 · 2026-07-14 · 回应 bayan 09:38 定向「抄对 Megatron 上游而非 fp32 兜底」

## 参考源（上游新鲜度铁律,fetch 于 2026-07-14）

| 源 | 路径 | commit | 日期 |
|---|---|---|---|
| **NVIDIA/Megatron-LM** (`main`) | `megatron/core/ssm/gated_delta_net.py` | `872442adc71fbff72db46ca35674dff9e4dabe83` | 2026-06-30 |
| **fla-org/flash-linear-attention** (`main`) | `fla/ops/cp/{context,comm,chunk_delta_h}.py` + `README.md` | `ebf3a0cff2be3e6f2b2f99820b8fe4e28855ced0`（cp 目录末次改动 `16f4f94`） | 2026-06-23 |
| 本地待查 | `experimental/lite/megatron/lite/primitive/modules/gated_delta_net.py` | worktree HEAD | — |

## 结论摘要（先读）

**bayan 假设「本地 shared 的 ring 递推抄漏/抄错了 Megatron 上游某步」的前提不成立——但方向性判断（shared 不该用、default 才对）是对的,原因比「抄错」更根本:**

1. **Megatron 上游的 GDN CP 根本不用 ring。** 它用 **head-parallel all-to-all**(`tensor_a2a_cp2hp`/`hp2cp`):把「序列切片、全部 head」重排成「全序列、head 切片」,每个 rank 在**完整序列**上对自己那份 head 跑标准单卡递推,再 a2a 换回。**无跨 rank state 递推、无 ring、无 merge**⇒ 数值上与非 CP **逐位等价**(head 之间独立,累加序不变)。
2. **本地 `sharded` 模式也不手写 ring——它委托 FLA `fla.ops.cp`。** FLA CP 是**序列分片 + all-gather + merge**:每 rank 从零态算局部 (S_ext, M),all-gather 后按 `S ← M_j·S + S_ext,j` 跨 rank 链式重构初始态。数学上精确,但**累加序不同**(多出 transition 矩阵 M_j 连乘)⇒ bf16 舍入与非 CP 路径**不同**。
3. 所以本地没有一段「与上游 ring 逐行对得上的代码」可供对照——**上游 ring 不存在**。本地 shared 的数值发散不是 glue bug(纯前向已到 bf16 地板 ~8e-3),而是 **FLA 算法本身的累加序**与 rollout/非 CP 路径不一致,在 RL 长程放大成 step1 ppo_kl 220×。

**上游照抄的真解 = 把本地 CP 从「FLA 序列分片 merge」换成「Megatron head-parallel a2a」**(数值精确、显存优于 replicated 全复制),而非逐行修 FLA ring,也非 fp32 兜底。

## 三种 CP 策略并排

| 维度 | Megatron 上游 `gated_delta_net.py` | 本地 `sharded`(FLA `ops.cp`) | 本地 `replicated`(A/B 里的 default 臂) |
|---|---|---|---|
| 分片轴 | 先 seq 后 a2a→**head 分片,全序列** | **seq 分片** | 全 gather→全序列全 head |
| 跨 rank state 数学 | **无**(head 独立) | **有**(merge `M_j·S+S_ext`) | 无 |
| delta_rule 调用 | `chunk_gated_delta_rule(cu_seqlens=cu_seqlens_q)` **不传 cp_context** | `chunk_gated_delta_rule(cp_context=...)` + `build_cp_context` | 本地 torch/FLA 全序列 |
| vs 非 CP 数值 | **逐位等价** | 累加序不同⇒bf16 噪声 | 逐位等价 |
| 显存 | seq×head/cp(优) | seq/cp(最优) | seq×full(最差) |
| 通信 | 2× a2a(cp2hp/hp2cp) | all-gather (S_ext,M) | all-gather 全张量 |

## 逐点代码对照

### 上游 Megatron(commit 872442a)—— head-parallel a2a,精确
- L24-26 `from megatron.core.ssm.mamba_context_parallel import _all_to_all_cp2hp, _all_to_all_hp2cp`
- L350-384 CP>1 时 `_build_head_perm_for_split_sections` + `qkvzba = tensor_a2a_cp2hp(qkvzba, seq_dim=0, head_dim=-1, cp_group)`：seq 分片 → head 分片、**全序列**。
- L472-482 `self.gated_delta_rule(query,key,value,g=g,beta=beta, initial_state=None, output_final_state=False, cu_seqlens=cu_seqlens_q)` —— **未传 cp_context**,即在全序列上跑普通 FLA `chunk_gated_delta_rule`,与非 CP 完全同一 kernel/同一累加序。
- L485-508 `tensor_a2a_hp2cp` 换回 seq 分片。
- L593-617 约束:每序列长度须整除 cp_size。

### 本地 `sharded`(`experimental/lite/.../gated_delta_net.py`)—— 委托 FLA ring
- L34-49 `from fla.ops.gated_delta_rule import chunk_gated_delta_rule` + `from fla.ops.cp import build_cp_context`。
- L135-150 `sharded` 分支:`_cp_swap_qkvzba`(zigzag↔contiguous all_to_all_single 重排)+ `_build_cp_context`。
- L162-179 `_causal_conv1d(..., cp_context=cp_context)` 与 `_gated_delta_rule(..., cp_context=cp_context)` —— **传 cp_context**,走 FLA 的序列分片 all-gather+merge 路径。
- L187-191 换回。
- **对照结论:** 本地这几段只是「把张量喂给 FLA cp_context」的 glue;真正的跨 rank state 数学在 FLA `fla/ops/cp/chunk_delta_h.py` 的 merge 里,不在本地。Megatron 上游没有对应段落(它不进 cp_context 分支)。故「逐行比 ring」无对象。

### FLA CP 算法(`fla/ops/cp/README.md` §CP Architecture)
- all-gather+merge:`S=0; for j=(r-n_pre)..(r-1): S ← M_j·S + S_ext,j`。
- 每 rank 局部先算 `S_ext`(零初态累积)与 `M`(transition,I 起始逐子块 `M←M_[t]·M`)。
- 这套 merge 的矩阵连乘 = 与单卡顺序递推**不同的浮点结合序**,是 shared vs default 数值差的根源。

## 建议(交 bayan 裁,属策略选型非一行照抄)

**选型序(bayan 铁律:上游照抄 > fp32 兜底):**

- **A(荐)· 移植 Megatron head-parallel a2a**:把本地 `sharded` 的 FLA-cp_context 路径,换成上游 `tensor_a2a_cp2hp`/`hp2cp` + 全序列 `chunk_gated_delta_rule`(不传 cp_context)。数值精确=消 220× ppo_kl,显存优于 replicated。**代价**:需移植 `mamba_context_parallel` 的 a2a/head-perm/thd-perm 工具 + 重构 forward(~数百行,跨 rank a2a 语义要对齐 mlite ParallelState),且须 GPU proxy A/B 验 shared==default。属大改+架构选型,先请 bayan 定夺是否投。
- **B · 直接用 `replicated`(即 A/B 的 default 臂)**:已验数值精确、reward 无病;唯一代价是全序列复制显存。若长上下文显存够,**零改动即解用户痛点**。
- **C · 保留 FLA `sharded` 仅限显存极限场景**,接受其非位精确(不建议做 RL 训练侧 log-prob 源)。

**不推荐:** fp32 累加兜底(bayan 已明示仅 debug 用,不改累加序本质,治标)。
