# MLite QAT 跨框架调研与施工设计（v2）

## 结论与签发请求

建议批准一个**以 NVIDIA ModelOpt/Megatron 为实现内核、以 MLite 模型协议为接入面**的 QAT
工作线。首个可施工增量是 **NVFP4 W4A16**（BF16 activations）；W4A4、FP8 和 MXFP4
只能在同一配置/导出合同已经验证后逐项打开。不要把 verl 的 FSDP `QATLinear` 移植进
MLite：它是正确的 FP4 算法和 `compressed_tensors` 导出参考，但 MLite 的模型分块、TP/PP/EP、
checkpoint 与 vLLM 同步路径都与 verl 的 ModelOpt/Megatron 支路一致。

本报告是调研和设计产物；没有修改任何生产 Python 代码，也没有提交 GPU 作业。

## 可复核的新鲜度与方法

2026-07-21T11:58Z（UTC）完成 fetch/浅克隆；下表的 commit 是本报告的源码证据锚点。
所有源码阅读均来自固定 commit，而非网页摘要。

| 参考 | 固定版本 | 调研结论 |
| --- | --- | --- |
| `verl-project/verl` | `a07d3e32d2b40ec35e9514ebebd7361db4d6f554`（`origin/main`，已 fetch） | 同时具有 FSDP 原生 QAT 与应对标的 Megatron/ModelOpt QAT。 |
| `inclusionAI/AReaL` | `59c1043ac94d0c6ee15b5b7eec8efae3e3d87ef6` | 有 Megatron/TE FP8 参数和 blockwise FP8 加载/测试；未发现 QAT/STE、NVFP4 或 INT4 QAT 实现。 |
| `THUDM/slime` | `ea9819f88caa5e043eb8aea992b0969ffe79aa8e` | Megatron BF16 训练 + SGLang FP8 是默认生产路径；INT4 fake-QAT/STE 是 beta，通过环境变量切入。 |
| `NVIDIA-NeMo/NeMo-RL` | `9f701f069af4424f96d44901c4b7e505bb5a34d1` | 最接近的完整参照：Megatron + ModelOpt QARL，支持 fake-quant 与 refit 时的 real NVFP4 export。 |
| `NVIDIA/TensorRT-Model-Optimizer` | `7d5d3f904620289e76287db865307168e79d68a6` | QAT 引擎、Megatron plugin、NVFP4/MXFP4/FP8 实现和 checkpoint 状态合同的权威来源。 |

### 源码证据索引

下列是本设计引用的最小一手证据面，便于签发者复核，不把“有某个配置名”误判成完整 QAT：

- verl FSDP：[Linear 筛选/替换](https://github.com/verl-project/verl/blob/a07d3e32d2b40ec35e9514ebebd7361db4d6f554/verl/utils/qat/core.py)、[NVFP4 fake-Q 与 STE](https://github.com/verl-project/verl/blob/a07d3e32d2b40ec35e9514ebebd7361db4d6f554/verl/utils/qat/linear.py)、[compressed_tensors pack](https://github.com/verl-project/verl/blob/a07d3e32d2b40ec35e9514ebebd7361db4d6f554/verl/utils/qat/quantizer.py)。
- verl Megatron：[chunk 接入](https://github.com/verl-project/verl/blob/a07d3e32d2b40ec35e9514ebebd7361db4d6f554/verl/utils/modelopt/qat_utils.py)、[W4A16 ModelOpt config](https://github.com/verl-project/verl/blob/a07d3e32d2b40ec35e9514ebebd7361db4d6f554/verl/utils/modelopt/quantize.py)、[PP/EP metadata 与 iterator export](https://github.com/verl-project/verl/blob/a07d3e32d2b40ec35e9514ebebd7361db4d6f554/verl/utils/modelopt/qat_weight_exporter.py)。
- NeMo-RL：[QARL guide](https://github.com/NVIDIA-NeMo/NeMo-RL/blob/9f701f069af4424f96d44901c4b7e505bb5a34d1/docs/guides/quantization-aware-rl.md) 明确规定 fake-quant policy backward 与 fake/real rollout 两种模式，并分别列出 W4A16/W4A4 的验证范围与风险。
- AReaL：[FP8 配置](https://github.com/inclusionAI/AReaL/blob/59c1043ac94d0c6ee15b5b7eec8efae3e3d87ef6/docs/en/cli_reference.md) 和 [blockwise FP8 export](https://github.com/inclusionAI/AReaL/blob/59c1043ac94d0c6ee15b5b7eec8efae3e3d87ef6/areal/engine/megatron_utils/fp8/quantize.py) 是 FP8 参数/导出参照；该固定版本源码检索没有 QAT/STE/NVFP4/INT4 QAT 实现。
- slime：[low-precision guide](https://github.com/THUDM/slime/blob/ea9819f88caa5e043eb8aea992b0969ffe79aa8e/docs/en/advanced/low-precision.md) 将 INT4 STE/fake-QAT 标为 beta，并用显式 flag 与 group size 开启；它不提供可直接复用的 ModelOpt NVFP4 导出协议。
- ModelOpt：[module quantization 入口](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/7d5d3f904620289e76287db865307168e79d68a6/modelopt/torch/quantization/model_quant.py)、[clipping STE](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/7d5d3f904620289e76287db865307168e79d68a6/modelopt/torch/quantization/tensor_quant.py) 与 [quantizer-state 存取](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/7d5d3f904620289e76287db865307168e79d68a6/modelopt/torch/quantization/utils/core_utils.py)。

外部规范/论文也做了网页核验：NVFP4 的 E2M1 + block-16 FP8 scale + tensor FP32
global scale 定义见 [Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)；MXFP4 是 OCP 定义的可交换 microscaling 格式，不能和 NVIDIA 专属 NVFP4 名称或 scale 布局混用，见 [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

## 三个状态必须分离

```
bf16 Parameter W  ── fake_quant + STE ──>  W_hat / X_hat  ──> forward loss
       │                        │                         │
       │                        └─ amax / scale buffers   └─ dL/dW_hat
       └──────── optimizer updates bf16 W <──────────────────── STE backward

export snapshot: bf16 W + frozen quantizer metadata ──> packed NVFP4 + scales
```

1. **训练参数（所谓 master weight）**：QAT 的可训练参数必须保留为原模型的 BF16 参数；
   `W_hat` 只是本次 forward 的 Q/DQ 结果，绝不可登记成 optimizer 参数或 checkpoint 的唯一
   权重。这里的“master”是量化域相对于连续权重的主副本，不是强制 FP32；若当前 optimizer
   本来另有 FP32 master/moments，仍由 optimizer primitive 管理，QAT 不拥有它。
2. **fake-quant/STE**：每个选中的 linear 在 forward 对 BF16 权重（W4A16）和可选 activation
   （W4A4/FP8）做 Q/DQ；反向仅使用 STE（必要时带 clipping）。量化 scale/amax 不是权重副本。
3. **部署表示**：只在 `export_weights`/rollout refit 创建 packed `uint8` FP4 与 scale 张量；训练
   step、optimizer、分布式 checkpoint 都不消费 packed 权重。

这与 ModelOpt 的 `TensorQuantizer._fake_quantize`/`_real_quantize` 分支一致，也与其文档所述
“量化后冻结 quantizer state、微调原权重”的 QAT 模型一致。LSQ 是未来可选的可学习 scale；
不是 v1 的默认行为。其理论依据是 [Jacob et al.](https://arxiv.org/abs/1712.05877) 的 affine-Q/DQ、
[LSQ](https://arxiv.org/abs/1902.08153) 的可学习 step size、[PACT](https://arxiv.org/abs/1805.06085)
的可学习 activation clip，以及低比特 LLM 的 [LLM-QAT](https://arxiv.org/abs/2305.17888)。

## verl 两条路径的事实核对

### FSDP 原生支路：算法/格式参考，不是 MLite 接入点

`utils/qat/core.py` 遍历 `nn.Linear`，跳过 `lm_head`、router 等不适配层，把符合 group-size
的层替换为 `QATLinear`。`utils/qat/linear.py`：

- 保留 clone 后的原 dtype 参数；`STEFP4QuantTriton.backward` 返回原样 `grad_output`。
- NVFP4 fake quant 使用 FP4 E2M1（最大幅值 6）、block size 16、FP8 E4M3 block scale，以及
  `global_amax/(6*448)` 的全局 scale。W4A16 只假量化 weights；W4A4 还维护 `input_amax` 与
  `input_global_scale`，可全局 all-reduce amax。
- `utils/qat/quantizer.py` 使用 `compressed_tensors` 的 `NVFP4PackedCompressor`，导出
  `weight_packed`、`weight_scale`、`weight_global_scale`，W4A4 另有 `input_global_scale`；并对
  QKV / gate-up 共享 global scale。

所以它验证了格式和 STE 思路，但 FSDP module replacement 不应越过 MLite 的 model/primitive
边界直接进入各模型。

### Megatron/ModelOpt 支路：MLite 的直接参照

`utils/modelopt/qat_utils.py` 在**每个 Megatron model chunk**建好后调用
`apply_qat_to_modules`，而不是在 runtime 里识别模型。`quantize.py` 的 W4A16 config 是
`weight_quantizer: num_bits=(2,1), block_sizes={-1:16,type:dynamic,scale_bits:(4,3)}`，关闭 input
quantizer，然后调用 `mtq.quantize(model, config)`；ModelOpt 负责替换/patch QuantModule 与量化器。

`qat_weight_exporter.py` 从 ModelOpt quantizer 读取 `amax` 和 block size，以 bridge 的 HF↔Megatron
mapping 找到每个已导出的 BF16 权重，产生：packed weight、`weight_scale`（FP8 block scale）、
`weight_scale_2`（FP32 global scale），有 activation quantizer 时额外产生 `input_scale`。它还在 PP
与 EP group 上汇总元数据。这正是 MLite export 要保持的“纯 iterator decorator”形态。

## 横向比较

| 框架 | QAT / 格式 | 后端与切点 | 与 verl/MLite 的关系 |
| --- | --- | --- | --- |
| verl FSDP | 自定义 NVFP4 W4A4/W4A16；Triton fake-Q + STE；`compressed_tensors` pack | `nn.Linear` 替换；FSDP 模型内 | 用作算法、scale 和 vLLM 格式参照。 |
| verl Megatron | ModelOpt NVFP4 W4A16（文件当前仅公开这个简化 config） | 模型 chunks 建立后；export iterator 前 | **直接结构参照**。 |
| NeMo-RL QARL | ModelOpt NVFP4 W4A4/W4A16；也有 FP8 KV；fake 或 real rollout | HF→Megatron import post-wrap 量化；load 前恢复 quantizer state；Megatron-Bridge refit/export | **端到端行为参照**：训练 fake quant，real rollout 每次同步 packed weights/scales。其文档明确提示通用 W4A4 GRPO 有收敛风险。 |
| slime | INT4 fake QAT/STE（beta）；FP8 training（实验）与 BF16→FP8 rollout（稳定） | Megatron/TE forward context；INT4 由 runtime env flag 启用 | 说明 QAT 需显式配置和模型级验证；不提供可复用的 ModelOpt export 合同。 |
| AReaL | 没有发现 QAT/STE/INT4/NVFP4；有 TE blockwise FP8 参数及 BF16 比对 | Megatron engine 的 FP8 模型/权重加载 | 只能作为 FP8 参数/加载校验辅助，不能作为 QAT 实现来源。 |

## MLite 落点与边界

MLite 的 `runtime` 只负责编排，`model` 负责 chunk 构建与 HF import/export，`primitive` 是可复用底层。
因此按 `primitive.design` 的 replaceability 与 `basic.find_reference` 的可检验合同，建议如下分层：

| 层 | 新职责 | 禁止承担 |
| --- | --- | --- |
| `primitive/quantization/qat.py`（新） | `QATSpec`、ModelOpt config adapter、对单 module 的 apply、checkpoint quantizer-state helper、导出 metadata reader | 不知道 Qwen/GLM/Kimi 名称，不调用 rollout。 |
| `model/*/lite/protocol.py`（按模型） | 在 chunk 创建、HF load 后、optimizer 构建前对 `chunks` 调 `apply_qat_to_chunks`；将 QAT export decorator 接到该模型既有 HF mapping iterator | 不复制量化算法或把状态藏进 runtime。 |
| `model/*/lite/checkpoint.py` | 以现有 HF 名称映射包装 export iterator；处理该模型的 fused QKV/gate-up 和 expert 名称 | 不直接修改 BF16 parameter 或 optimizer state。 |
| `runtime/backends/mlite` | 只传递 typed QAT config，保留现有 `export_weights` 生命周期 | 不按 module 名称做替换、不 import ModelOpt。 |

`QATSpec` 应是显式 opt-in（默认 `enabled=False`），最小字段：`format`（首发仅
`nvfp4_w4a16`）、`weight_block_size=16`、`calibration`（max/冻结策略）、`ignore_patterns`、
`export_mode`（fake/real）、`learnable_scales=False`。`format` 不可接受自由字符串；每个枚举都要有
精确的 ModelOpt config 和 export schema。FP8 是另一枚举/recipe，不可把 FP8 scale 当 NVFP4 scale。

**W4A4 的额外门槛**：先有校准/observer 冻结协议、跨 DP amax 同步和 `input_scale` export，再允许打开；
不因可运行而宣称收敛。**MXFP4 的额外门槛**：它遵循 OCP MX（通常 E8M0 block scale）而 NVFP4
使用 E4M3 + FP32 global scale，必须另建 serializer 和 vLLM compatibility test，不能复用 NVFP4
tensor 名称或校验阈值。

## checkpoint、并行与 rollout 合同

- checkpoint 必须保存/恢复 BF16 parameters、optimizer state 及每个 quantizer 的完整 state（amax、
  scale、enabled/fake mode、LSQ 参数若启用）。恢复顺序为：build chunks → apply quantizers → 恢复
  quantizer metadata → 加载模型/optimizer；否则 state dict key/shape 会漂移。
- TP 的量化统计必须明确 shard-local 或 group-reduced；PP/EP export 元数据需要 ModelOpt/verl 同样的
  group 汇总，不能只从本 rank 推断全局权重。目标模型的 fused QKV/gate-up 与 expert packing 走已有
  model mapping 后再 quantize，避免 primitive 了解模型命名。
- `export_weights` 的输入和输出仍是 `(hf_name, tensor)` iterator。训练状态没有 packed weight；每次
  rollout refit 从当前 BF16 snapshot 生成 packed tensor+scale。输出命名/quantization config 必须符合
  vLLM `compressed_tensors` 的 `nvfp4_pack_quantized` schema。该格式对 NVFP4A16/W4A4 的分派要求可由
  [vLLM compressed-tensors 文档](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors/) 核验。

## 施工顺序、验收与不可签发项

1. **零 GPU、TDD**：新增 QAT config 解析和最小 toy chunk 测试。验证 selected modules 获得 ModelOpt
   quantizer、BF16 `Parameter` identity/dtype 不变、forward fake-Q 实际执行、STE 后 `weight.grad` 非空；
   disabled config 必须 bit-identical 不插入节点。
2. **checkpoint 合同**：CPU/单卡 proxy 验证 save→fresh build→restore 后 quantizer state、BF16 参数和
   下一步 optimizer 行为一致。覆盖 PP/EP metadata 的纯对象单测；真正分布式行为留给 Slurm。
3. **export 合同**：固定小权重与 verl/ModelOpt reference 逐字节比较 packed values、scales、名称和
   order；W4A16 再跑 vLLM/`compressed_tensors` load。遵守既有 K-0135/K-0153/K-0150：真实 quant-HF
   dequant、BF16 直通、quant→export-BF16→清零→reload 的 maxdiff=0/参数完整性证据不可省略。
4. **Slurm GPU 门**：先 1 GPU dense toy 的 BF16/QAT loss 与梯度有限性，再 8 卡缩尺 TP/EP/PP proxy，
   最后才申请实际模型。W4A4、FP8、MXFP4 各自重新通过该链路；不共享“W4A16 已过”结论。
5. **科学验收**：同 seed、数据、步数、学习率下报告 BF16 baseline、fake-Q train/eval、real-Q rollout
   三者的 loss/reward/throughput/显存；W4A4 需要独立收敛判定。研究文献只说明方法可行，不构成阈值；
   具体阈值须由任务 owner 签发。

本次不请求签发的内容：默认打开 QAT、FP32 master 强制转换、W4A4 收敛承诺、MXFP4 支持、将 packed
权重写入训练 checkpoint，或任何 GPU 结果。它们都需要后续独立施工节点和上述证据。

## MLite skills 对照

- 已按 `basic.find_reference` 选择 ModelOpt/Megatron 为可检查主参考，verl FSDP 为交叉参考；报告给出
  dtype、shape、format、导出名字和变量冻结合同，而非只列 API。
- 后续实现必须遵守 `primitive.design`：quantizer primitive 可替换、没有模型名称依赖；model protocol
  显式组合；runtime 不下沉 module knowledge。
- 若改动模型配置/出口，继续加载 `model_compose.config_mapping` 与 `model_compose.weight_mapping`；若改动
  checkpoint，加载对应 checkpoint skill。任何新增/修改 MLite skill 才需运行 `basic.lint_skill`，本报告
  未改 skill 文件。
