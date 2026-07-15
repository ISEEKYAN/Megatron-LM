# DS4 fp8 resync · logit-similarity 动态测设计（TASK-1.1.12 拆桥验证 step-3，待审）

**定位**：验证阶梯第 3 层（决定性）。前两层已绿：
1. ✅ CPU block-fp8 round-trip（quant 数学，expert 0.33%/0.35%）— `test_block_fp8_roundtrip.py`
2. ✅ 静态布局对照表（无 transpose/merge，加载不崩佐证）— `ds4_fp8_resync_layout_audit.md`
3. ⏳ **本设计**：锁残留唯一值级点 = **block-scale 逐块对位**（每块 scale 是否贴对同一块权重；shape 相同不保证逐块对位，静态验不了）。

**原理**：同一批 token 分别过 ①vLLM rollout（fp8 resync 后权重）②mlite 训练侧 forward（bf16 actor 权重），比**输出头 logits**。block-scale 逐块对位对 → 两侧 logits 高度相似（仅 fp8 量级差）；对位错 → logits 发散（错块 scale = 权重乱 = 输出乱）。

---

## ① 怎么减层装 8 卡
- **改 `num_hidden_layers`**：override 到 `first_k_dense_replace + 2`（= 全部初始 dense 层 + 2 层 MoE）。**必须含 ≥1 真 MoE 层**——block-scale 对位只在 fp8 量化的 routed-expert 权重上有意义，纯 dense 层验不到。
- **保留真权重**：两侧都从真 DeepSeek-V4-Flash checkpoint 载**前 N 层**的真实权重（embed/head/norm 全尺寸真值），忽略其余层。**非随机**——验的是真权重经 fp8 往返后数值对不对。
- **单节点 8 卡，PP1**（避开 PP/THD packing，属别的 worker）：mlite forward 用 PP1 + TP/EP 装下减层模型；vLLM rollout_tp=8。减层后模型小，8 卡宽松。
- **专家激活覆盖**：block-scale 是 per-expert per-block；只验到被激活的专家。用**足够多样的 token 批**（如 ≥64 prompt）让尽量多专家路由到；**报告专家激活覆盖率**（激活了 E 个中的几个），未激活专家标注"本测未覆盖"。

## ② 怎么抓两侧同 token 的输出头 logits
- **固定确定性 token 批** B（如 64 prompt × ~128 token，seed 固定，落盘复用），两侧喂**完全相同**的 input_ids。
- **mlite 侧**：对 B 做一次 **teacher-forcing forward**（bf16，无采样），抓 `lm_head` 输出 logits `L_mlite[B, T, V]`（或每位置 realized-next-token 的 logprob）。落盘。
- **vLLM 侧**：对 B 用 `prompt_logprobs=K`（如 K=20）+ `max_tokens=1`，vLLM 对每个 prompt 位置返回 top-K logprob **且总含 realized token 的 logprob**。抓 `L_vllm`。落盘。
- **对齐**：按 (prompt_id, position) 对齐；vocab 维在两侧 top-K 的**并集支撑**上比（vLLM 非全 vocab）；realized-token logprob 恒可比（两侧都有）。

## ③ similarity 阈值（对位对 vs 错，差距极大，阈值好放）
逐位置计算、跨批聚合：
- **主判据 A**：next-token **top-1 argmax 一致率**（mlite argmax vs vLLM top-1）。**≥95% = 对位对**；对位错时坍到 ~随机（<50%）。
- **主判据 B**：每位置 logit 向量（top-K 并集支撑）**cosine 相似度**均值 **≥0.98**；对位错 <0.9。
- **辅助**：realized-token logprob 的 Pearson 相关 **≥0.99**；`KL(vLLM‖mlite)` 均值 **<0.1 nats**。
- **判决**：A≥95% 且 B≥0.98 = **block-scale 逐块对位对 = 拆桥数值全对** ✅；A 坍塌/B<0.9 = **对位错**（报 bayan 别硬修）。
- **容差依据**：vLLM 本身跑 fp8（权重 + kv_cache fp8）→ 与 bf16 mlite 有内生 fp8 噪声（几 % 相对），故阈值留 fp8 量级容差；但对位错=错块 scale=权重量级级乱→logits 灾难性发散（~随机），与"正确 fp8 噪声"(~99%)差一个数量级，阈值区分度极高。

## ④ 8 卡怎么同时起 vLLM + mlite forward
**复用现成 debridge verl colocated 载具**（mlite actor bf16 + vLLM rollout fp8-resync 同进程，就是我们已验的 resync 路），加一个 **post-resync 诊断钩子**（env-gate `MLITE_LOGIT_SIM_PROBE=1`，DS4-only）：
1. 首次 resync 完成后（fp8 hook 已跑），触发探针；
2. 读固定 token 批 B；
3. 调 mlite actor forward(B) 抓 `L_mlite`（走 verl 已有的 `compute_log_prob`/forward，teacher-forcing，不经 update_actor）；
4. 调 vLLM engine `generate(B, prompt_logprobs=K, max_tokens=1)` 抓 `L_vllm`；
5. 两侧落盘 + 算 ③ 的指标 + 打印判决；
6. 探针后**直接退出**（不进训练步，天然避开 packing bug）。

> 用 colocated 载具的关键好处：vLLM 侧权重是**真经 resync 路来的 fp8**（非旁路重载），验的是整条拆桥产出，最faithful。

---

## 执行阶梯 + 红线
出设计 → **bayan 审** → 建减层载具（fork debridge 8卡 smoke，改 `num_hidden_layers` + 加探针）→ **8 卡 fire（待 bayan，GPU 铁律不自烧）** → 出两侧 logits 样本 + 指标 + 判决。
**不碰 AC#3（Megatron worker）、不修 packing bug（1.1.12.8 worker）、不发 128。**

## 待审口径（3 点）
1. **减层数**：`first_k_dense_replace + 2` MoE 层够否？还是要更多 MoE 层提高专家覆盖（代价=更大，仍单节点）。
2. **logits 抓法**：mlite 走 verl 已有 `compute_log_prob`(teacher-forcing) vs 直接 hook `lm_head` 输出全 logits——前者省事复用、后者拿全 vocab。倾向前者 + realized-logprob 为主指标。
3. **载具**：post-resync in-process 探针（推荐，最faithful）vs 两阶段落盘离线比（mlite dump→vLLM dump→offline）。倾向前者。
