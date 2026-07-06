# TASK-1.1.15 Gate0 版本与配置对齐记录

更新: 2026-07-05(驻场秘书 TASK-1.1.15.1)。集群定为 **cw/H100**(bayan 裁定,不用 H20)。

## 已锁定

| 项 | 值 | 来源 |
| --- | --- | --- |
| 本轮 mlite worktree HEAD | `c178e1c3e` (ISEEKYAN/Megatron-LM, feature/debug-rl-bayan-debug-manual) | 本 worktree `git log` |
| 旧受控基线 MLite | `41e752596` | TASK-2.16.5.15.2 归档记录 |
| 旧受控基线 VERL | `c3ef4275` | 同上 |
| 容器(计划首选) | `verl.vllm023.sqsh`(~29GB, 容器内 `/usr/bin/python`, PYTHONPATH 需 `/vllm` 打头) | llmrl-experiment-runbook + 归档 |
| vllm023 疑似绝对路径(待 cw stat 确认) | `/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/verl_optimize/verl.vllm023.sqsh` | 归档任务记录 |
| cw 控制链路 | llmrl-launch doctor 全绿(ssh/venv/protocol/snapshot-root, 2026-07-05) | 本机实跑 |

## 待确认(cw 只读探测)

- [ ] `verl.vllm023.sqsh` stat + sha256
- [ ] cw 上 verl 仓 HEAD(`$U/code/megatron_lite/verl`)与旧基线 c3ef4275 的差距
- [ ] cw 上 Megatron-LM 仓 HEAD 与本轮 c178e1c3e 对齐方式
- [ ] Qwen3.5-35B-A3B 模型目录(含 tokenizer/config 完整性;报告方要求 max_position_embeddings=32768)
- [ ] dapo-math-17k.parquet / aime-2024.parquet 数据路径
- [ ] 报告方(原 H20 run)的 verl/mlite/vllm commit 与镜像——向报告方索取
- [ ] Megatron 基线原始 script(源 script 中 MEGATRON_ACTOR 已删)——向报告方索取
- [ ] 两侧 resolved hydra config diff(G0b DRY_RUN 产出)
- [ ] fresh/no-resume 证据(CKPTS_DIR 全新)

## 已发现的配置疑点(gate0 必查)

1. **calculate_log_probs 死分支**:源 script `using_rs=False` 时 RS_CONFIG(含
   `rollout.calculate_log_probs=True`)整组不透传——与历史 F3 坑(old_log_prob=None)
   同型。移植版已改为无条件透传;需在 resolved config 里核实生效。
2. mLite grad_norm ~1/3-1/6 于基线:查 THD packing 下 loss 归一/mask/clip 口径
   (旧 TASK-2.16.5.2 修过同类)。
3. 旧战役已排除项(不必重查):权重同步(W0 全覆盖 max_abs=0)、GDN chunk↔fused_recurrent
   (Δ0.5%)、cascade/no-chunk/cudagraph 配置项。已知固有 mismatch:layer0 GDN attention
   首差经 MoE 放大~28x,live decode cache 占 L1 幅度~40%(8K 响应下 diff_std~1e-2)。
   本轮 20K 响应 + CP4 是否放大到致崩,由 G2 短步 RL 判定。

## 复现脚本

`scripts/debug_rl/run_dapo_h100.sh`:双变体(BACKEND=mlite|megatron),超参与源
script 逐字段一致;差异仅 ①恢复 Megatron 基线数组 ②logprob 透传修复 ③路径参数化。
`DRY_RUN=1` 输出 resolved 命令供 config diff。
