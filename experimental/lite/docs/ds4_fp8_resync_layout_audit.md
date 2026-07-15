# DS4 fp8 resync 静态布局对照表（TASK-1.1.12 拆桥数值验证 step-2）

**目的**：验证 mlite fp8 导出的**布局/轴序/融合/scale 布局**与 vLLM Fp8MoEMethod 期望**逐项对位**，
抓 transpose（轴序反）、merge 差异（gate/up 融合）、`weight_scale_inv` 命名/shape 不匹配。
（CPU round-trip 只证 quant 数学；本表证布局；logit-similarity 是决定性动态确认。）

## 三侧代码坐标
- **mlite 导出**：`megatron/lite/model/deepseek_v4/lite/resync.py::export_resync_weights`
  + `checkpoint.py`（naming：fc1→w1/w3、fc2→w2，L186-190）+ `block_fp8.py::quantize_block_fp8`
- **verl 中间 mapper**：`verl/utils/vllm/vllm_dsv4_fp8_utils.py`
  （`iter_deepseek_v4_weights` L35；`_map_weight_name_for_vllm` L273）
- **vLLM 期望**：`vllm/model_executor/layers/quantization/fp8.py::Fp8MoEMethod.create_weights` L534
  + `vllm/models/deepseek_v4/*/model.py::load_weights`（`fused_moe_make_expert_params_mapping` ckpt_gate="w1"/ckpt_up="w3"/ckpt_down="w2"）

> 记号：E=专家数, H=hidden_size, I=moe_intermediate_size, 块=(128,128)。轴序 = **[out, in]**（标准 nn.Linear）。

## 对照表（每专家；vLLM 侧为融合后 [E,...] 的对应切片）

| 项 | mlite 导出 | verl 改名后 | vLLM 期望 | 匹配? |
|---|---|---|---|---|
| gate 权重 | `…experts.{E}.w1.weight`  `[I,H]` `float8_e4m3fn` | 同名（+`model.`前缀） | `w13_weight[E, 0:I, :]` = `[I,H]` `e4m3`（shard "w1" off 0） | ✅ 名(w1=gate 两侧同义)/轴[I,H]/无转置 |
| up 权重 | `…experts.{E}.w3.weight`  `[I,H]` `e4m3` | 同名 | `w13_weight[E, I:2I, :]` = `[I,H]`（shard "w3" off I） | ✅ 融合由 vLLM narrow-copy 做，我们**正确不预融合** |
| down 权重 | `…experts.{E}.w2.weight`  `[H,I]` `e4m3` | 同名 | `w2_weight[E,H,I]` `e4m3` | ✅ 轴[H,I]/无转置 |
| gate/up scale | `.w1.scale`/`.w3.scale` `[⌈I/128⌉,⌈H/128⌉]` `float32` | `.w1.weight_scale_inv`… | `w13_weight_scale_inv[E, 2⌈I/128⌉, ⌈H/128⌉]` `f32` | ✅ 块网格行 I/128 堆叠成 2I/128 |
| down scale | `.w2.scale` `[⌈H/128⌉,⌈I/128⌉]` `f32` | `.w2.weight_scale_inv` | `w2_weight_scale_inv[E, ⌈H/128⌉, ⌈I/128⌉]` `f32` | ✅ |
| 权重 dtype | `float8_e4m3fn` | 直通（fp8 experts 不 `.view(uint8)`，L37 仅对 int8/e8m0） | `float8_e4m3fn` | ✅ |
| scale dtype | `float32`（`scale_format="float32"`，fp8 分支） | 直通 | block-quant `weight_scale_inv` = `float32` | ✅ |
| scale 命名 | `.scale` | `.scale`→`.weight_scale_inv`（L287-288） | `w13_/w2_weight_scale_inv` | ✅ 改名到位 |
| 名前缀 | `layers.…` / `head.weight` | `model.layers.…` / `lm_head.weight`（L294-297） | vLLM 前缀 | ✅ |

## 判定：布局逐项匹配，**无 transpose / 无 merge 错**
- **无转置**：gate/up=[I,H]、down=[H,I] 两侧一致；且 I≠H，若有转置 vLLM 的 `weight_loader` narrow-copy 会在**加载时 shape 崩**——8卡(job14026335)+128卡(job14028583)**均无 shape/narrow 崩**、`process_weights_after_loading` 在 43 专家跑通 = **shape/轴/融合/scale-shape 合同结构性满足**（强佐证）。
- **融合正确分工**：mlite 导**分开** w1/w3；vLLM 经 `expert_params_mapping(ckpt_gate="w1",ckpt_up="w3")` 自己 narrow 拼 w13。我们**不预融合**是对的（预融合反而会与 loader 冲突）。
- **命名合同**：`.scale`→`.weight_scale_inv`、`w1/w3/w2`↔`ckpt_gate/up/down` 全对上；无孤儿/漏映射。
- 早先 “w13 2048 vs 4096” 是 **fp4 打包(2 值/字节 → H 减半)**，非转置——已由 `expert_dtype=fp8` 消解（fp8 不打包）。

## 静态验不了、须 logit-similarity 才能锁的残余点（值级，非 shape 级）
1. **gate/up 值序**：w1/w3 同 shape `[I,H]`，即使**互换**也不 shape 崩——名两侧都 w1=gate 结构上安全，但只有 logit-sim 能证没被换。
2. **block-scale 网格朝向**：`quantize_block_fp8` 的块索引朝向须与 vLLM 期望的 `[…/128]` 网格逐块对位；shape 相同不保证每块 scale 贴对同一块权重。
3. 上述都需 **③ logit-similarity（减层真权重 8卡，vLLM-fp8 vs mlite-bf16 同 token 输出头 logits 比对）** 决定性确认——布局若错，logits 必发散。

## 验证阶梯状态
1. ✅ CPU block-fp8 round-trip（quant 数学）——`test_block_fp8_roundtrip.py` 绿，expert 0.33%/0.35%、极值 2.55% < 6%。
2. ✅ 本静态布局对照表——逐项匹配，无 transpose/merge 错（+ 加载不崩强佐证）。
3. ⏳ logit-similarity（决定性动态）——设计待审。
