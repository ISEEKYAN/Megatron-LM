# TASK-1.1.15 Gate0 版本与配置对齐记录

更新: 2026-07-05(驻场秘书)。集群定为 **cw/H100**(bayan 裁定,不用 H20)。cw 探测经 `cwssh` wrapper(已入 allowlist)。

## 已确认(cw 实测 2026-07-05)

| 项 | 值 |
| --- | --- |
| 容器 | `$U/code/verl_optimize/verl.vllm023.sqsh` 存在,28,903,591,936B,mtime 2026-06-26(vLLM 0.23.1/Py3.12/TE 2.15 内置 moe_permute+rope patch,免 bind-mount) |
| cw mlite HEAD | `c178e1c3e` == 本 worktree HEAD(已对齐) |
| cw verl HEAD | `7cf31bf9`(megatron_lite/verl);旧受控基线为 `c3ef4275`——**pin 待 bayan 定**(建议沿用 AB 时的 verl-main-latest) |
| 模型 | `/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B`(14 shards;config `max_position_embeddings=262144`,报告方 script 要求改 32768——按本轮 response 22K 上限其实 32768 够,是否改随 AB 老配置) |
| 数据 | `$U/code/verl_update_mcore/data/{dapo-math-17k.parquet(299MB), aime-2024.parquet}` 存在 |
| **proven A/B 全套** | `$U/code/qwen35_dapo_ab/`:AB_REPORT.md + megatron_proven.{sh,sbatch,node.sh} + mlite.sbatch/mlite_node.sh,统一 env=vllm023+verl-main-latest+mcore 0.19.0dev+mbridge main+CP overlay(qwen35-cp-overlay-20260613/site) |

`$U = /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan`

## 老 AB(TASK-2.16.5, 2026-06-27)关键结论——本轮排查的起点

- step1 rollout/reward/advantage 侧高度对齐(score -0.828 vs -0.831);**未解决遗留 = grad_norm/pg_loss 口径不对齐**:
  MG grad_norm 平坦 0.05-0.07,ML 从 0.134 单调衰减到 0.02;MG pg_loss==|adv_mean|,ML≈一半。
  怀疑方向(AB_REPORT §5):token-mean 分母(local vs global)、micro-batch 累加归一、grad_norm 计算来源、DP/EP grad reduce 语义。
- **这与本轮 H20 报告"mlite grad_norm 明显偏低"直接吻合——假设2(loss/grad 口径)升为首位**。
- 代码位置:verl `workers/engine/megatron/transformer_impl.py` L607/L757/L835-905;mlite engine=experimental/lite;共同 loss=`verl/trainer/ppo/core_algos.py`。

## 报告方(H20) script 与老 AB 配置的差异(G2 复现前要决定取哪套)

| 项 | 老 AB(proven) | 报告方 H20 script |
| --- | --- | --- |
| bs/gen_bs | 32/96 | 256/32(且 gen_prompt_bsz=mini_bs,存疑) |
| max_response | 8192 | 20480 |
| mesh | TP1 PP1 CP1 EP8 | TP1 PP1 CP4 EP8 |
| offload | (见 proven 脚本) | param/opt/grad 全 offload+recompute full |
| calculate_log_probs | (查 proven 脚本) | **藏在 using_rs=False 死分支,实际未透传** |

## 待办(gate0 收尾)

- [ ] bayan 定 verl pin(cw HEAD 7cf31bf9 vs 老基线 c3ef4275 vs AB 的 verl-main-latest 快照)
- [ ] 读 megatron_proven.sh/mlite_node.sh 全文,提取 resolved 超参作为 G2 的 A/B 基准配置
- [ ] 向报告方索取:H20 侧 verl/mlite/vllm commit + Megatron 基线原始 script
- [ ] G0b DRY_RUN 双侧 resolved config diff

## 复现载具决策(建议)

优先复用 `qwen35_dapo_ab` proven 管线(gate0 对齐成本≈0,只需把 mlite worktree 推到 c178e1c3e),
本 repo `scripts/debug_rl/run_dapo_h100.sh`(报告方 script 的移植版)作为"复现报告方现象"的对照载具。
