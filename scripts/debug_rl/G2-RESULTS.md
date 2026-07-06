# G2 短步 RL A/B 结果(round4, 2026-07-06)

Job: mlite=13483436 / megatron=13483439(step3 后按早停原则 scancel;megatron 基线 13485548 重提跑满 10 步)。
条件:vllm023 容器、verl-main-latest a6ef5009、同 mesh TP1PP1CP1EP8、同数据/模型、fresh/no-resume、
**双侧 betas=[0.9,0.999]、双侧 calculate_log_probs=True**、TRITON_CACHE_DIR=/tmp(runtime_env 注入)。
mlite 引擎=wt-qwen35-dapo@a5d035b26。

## 结论:分叉复现(排除 betas/环境污染后依然存在)

| step | grad_norm ml/mg | 比值 | pg_loss ml/mg | ppo_kl ml/mg | score ml/mg |
|---|---|---|---|---|---|
| 1 | 0.1334 / 0.0411 | 3.2x | 0.0076 / 0.0202 | 5.1e-4 / 7.8e-6 | -0.844 / -0.831 |
| 2 | 0.1316 / 0.0338 | 3.9x | 0.0127 / 0.0108 | 7.1e-4 / 1.3e-5 | -0.914 / -0.854 |
| 3 | 0.1186 / 0.0461 | 2.6x | 0.0043 / 0.0181 | 6.4e-4 / 1.2e-5 | -0.862 / -0.805 |

- rollout/reward/advantage 侧对齐:score、advantages/mean(s1: -0.0195/-0.0209)、response_length(~8.0K)、
  step0 AIME acc(0.1115/0.1031)均一致 → 分叉在训练更新侧,与老 AB 结论一致。
- megatron 复现老 AB 形态:grad_norm 平坦 0.03-0.05,**pg_loss≈|advantages/mean|**(s1 0.0202 vs 0.0209,
  s3 0.0181 vs 0.0187);mlite pg_loss 无此关系且波动大。
- **新线索:ppo_kl 差 40-65 倍**(mlite ~6e-4 vs megatron ~1e-5)。ppo_kl≈KL(old_log_prob‖训练 forward logprob),
  说明 mlite 的训练 forward 与其 old_log_prob 来源之间存在远大于 megatron 的系统性偏差——把
  「grad_norm 口径」与「H20 报告的 k3_kl/chi2 升高」两条异常连到了同一个嫌疑面:
  mlite 训练侧 forward(或 old_log_prob recompute 路径)与生成分布不一致。
- 与 H20 报告方向差异:报告里 mlite grad_norm 偏低,本轮(与老 AB 同向)偏高——两者配置不同
  (CP4/20K/bs256 vs CP1/8K/bs32),方向可翻转但"口径不一致"这一异常本身两处都在。

## 下一帧(诊断,建议派 worker)

四个既有怀疑点(老 AB §5)+ 新增 ppo_kl 线索:
1. token-mean 分母:local micro-batch token 数 vs 全局 token 数(core_algos agg_loss vs 两 engine 的聚合)
2. dynamic_bsz 下 micro-batch 梯度累加/平均口径(megatron get_forward_backward_func vs mlite train_step.run_microbatch_loop)
3. grad_norm 计算来源与 reduce 范围(megatron optimizer clip_grad_norm(DP all-reduce 后) vs mlite fsdp2 torch clip;EP 参数组)
4. DP/EP 梯度 reduce 语义(mean vs sum)
5. 【新】old_log_prob→训练 forward 的 logprob 偏差:mlite compute_old_log_prob(forward-only)与 train forward
   是否走了不同路径(fused_kernels/THD packing/recompute);量化 per-token logprob diff 分布
代码入口:verl `workers/engine/megatron/transformer_impl.py` L607/757;mlite `verl_mlite/engine/mlite_engine.py`
forward_backward_batch L361/L675 + `megatron/lite/primitive/train_step.py` run_microbatch_loop L57;
共同 `verl/trainer/ppo/core_algos.py`。
