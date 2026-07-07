# 夜间托管晨报(2026-07-06 夜,07-07 晨刷新)—— ⭐bayan 请先读这里

## 一句话结论(07-07 晨版,推翻昨晚 fsdp2 归约结论)
grad_norm 3.4x 根因**未闭环但大幅收敛**:昨晚"FSDP2 归约口径"已被实验推翻(lite+distopt 也 0.140,job13517351)。
确证事实链(全有 job id,详见 .10 notes 02:14 条):只发散在**参数 wgrad**(dgrad 对齐 1.007-1.02)、
只在**长序列 8289 触发**(SFT 短序列恒对齐,含 micro>1/recompute=full)、逐族越深越大(head1.0→GDN4.0)、
**head 是唯一对齐族且是唯一不走 TE 的层**(vanilla matmul,linear.py)——最新硬线索=**TE Linear wgrad 路径**(mlite 用 TE 的方式 vs MCore),具体行未定。
已排除(实测):优化器/FSDP2·norm 归约/loss 缩放(13516738 负结果)/expert1/8/THD 边界/micro 累加/recompute/六罪犯。

## 待 bayan 决策(2项)
1. **[方向裁定] grad_norm 收口路线**——二选一:
   - A) 继续钻 TE Linear wgrad:比对 mlite 调 TE Linear 的方式 vs MCore(wgrad 精度/fuse/accumulation 口径),有 12层分钟级 proxy+逐族比值口径可用,收敛在望但周期不定;
   - B) **转 10-step RL 看 acc 实效**(grad_norm 差记为已定界 backlog:wgrad×长序列×TE 路径;"RL 能 work"是终极交付标准)。
2. **[unblock] TASK-1.1.15.3**(10:10 起挂,15K 三方对比被卡):
   `env -u VICKY_ACTOR -u VICKY_TASK -u VICKY_LAUNCH_ID vicky unblock TASK-1.1.15.3 --as bayan`
   (若选 B,.3 正好用全修复 mlite 跑 15K 看 acc,一并解决决策1的验证)

> 注:以下"六罪犯/版本勘误"仍有效;昨晚流水里 12:22 后的 fsdp2/优化器方向结论已被 07-07 凌晨实验修正,以本节和 .10 notes(02:14/02:23 两条)为准。

## 六罪犯修复(独立成立,建议先落 PR)
1 聚合契约(PR#80已含) / 2 router BF16 / 3 GDN 编译契约 / 4 RoPE inv_freq FP32 / 5 export 重构丢失 / 6 dispatcher permute fusion
——全是 loss/精度侵蚀项,修复让同批 replay loss 对齐到 0.01%;上游 is/main 已覆盖 3-4 个(详见下)。

## 版本重大勘误
本战役 base c178e1c3e = 6-08 老 main,落后 is/main@69ea18d07(6-30)1468 commits;上游已修聚合/export/inv_freq/parity。
给报告方候选答案:升级 ≥69ea18d07 + 我方 router/GDN 补丁重跑。

---
## 夜间流水(倒序)

1. **unblock TASK-1.1.15.3**(10:10 headless 超时又挂 needs_human;mlite@is/main 15K 三方对比被它卡着):
   `env -u VICKY_ACTOR -u VICKY_TASK -u VICKY_LAUNCH_ID vicky unblock TASK-1.1.15.3 --as bayan`

## 主线(.9)夜间进展
- 10:28 ep8_depth proxy(13498384):EP8+全深度 **grad diff=0**——proxy 域穷尽,模型数学全面洗清
  (GDN/RoPE/router/recompute/EP8/深度全部单项无罪)。
- 推论収窄:3.3x 只活在**完整管线**,嫌疑=verl↔engine 集成层(dynamic_bsz 切批/micro 累加/
  fsdp2 offload DTensor 梯度/mini-step 间 zero_grad)。
- 已下发新实验令:完整管线内容敏感性判决 + micro 循环插桩(每 micro 梯度贡献/zero_grad 实况/
  loss_mask token 数)。**mlite 三个 mini-step norm 恒定 ±2% 的最简解释=梯度累积未清**,待插桩证实。

## 夜巡增量
- 10:50 EP8 layer0 分段(13498601):**首差=MoE 合成段,候选罪犯 6=shared expert 调用面**——
  mlite(非确定路径)走 Column/Row parallel wrapper,MCore 走 plain TE Linear;GDN o_proj/mlp_norm/
  routed experts 全 exact。追查 13499476 在跑。proxy 域并未穷尽——EP8+stage 粒度才暴露此差。
- .6 第 3 次 headless 超时挂 needs_human(10:40)——晨报队列 +1;建议改交互式 vicky run 或调大超时。

- 11:20 突破:40层全专家 EP8 proxy **复现强分叉**(迭代域夺回,分钟级可用);layer0 上 shared/routed
  expert 输出均逐字节一致——**首差收窄到 routed combine/unpermute/最终合并序(MoE TokenDispatcher)**,
  worker 在对照两侧 dispatcher 并插桩。罪犯 6 候选从 shared expert 改判 dispatcher combine 路径。

- 11:53 **罪犯6定罪+修复**:dispatcher permute fusion——mlite Qwen3.5 未显式启用 MoE dispatcher
  的 permute fusion(MCore 默认走),routed combine/unpermute 分叉源。修复 commit 1cd4573f3,红-绿+
  33项聚焦回归通过;全深度 EP8 终验 13500499 在跑。**六罪犯全部定罪**(1聚合/2router/3GDN编译/
  4RoPE/5export/6dispatcher fusion)。下一步:全模型同批 replay 比值终验→is/main port→10-step RL。

## ⚠ 12:22 反转警示
- 全模型 replay 13500928(dispatcher 修复后)grad_norm **仍 0.140=3.33x**,loss 仍对齐。
  => 六罪犯修复对 loss 有效,对全模型 grad_norm 比值**未见效**。dispatcher fusion 的单元级证据
  (grad_abs=0)可能是窄证明,不代表全模型梯度贡献占比。**grad_norm 3.3x 仍未真正闭环**。
- 已下强制单变量校验:fusion on/off 严格对照 + 查 effective 值是否被 launcher/env 覆盖(类 betas 坑)+
  嫌疑退回 verl↔engine 集成层(micro 累加/DTensor reduce)。晨报需 bayan 知悉:定罪≠已解决,勿轻信。

## 12:56 硬对照(秘书直接判读)
- 两发全模型 replay 天然构成 fusion on/off 对照:13497653(修复前)gn=0.140048 vs 13500928(修复后)
  =0.139983——**dispatcher 修复对全模型 grad_norm 无影响(0.05%)**。确证:六罪犯全修后 grad 仍 3.3x。
- **grad_norm 3.3x 主因至今未定位**;嫌疑集中 verl↔engine 集成层(micro 累加口径 / fsdp2 DTensor
  grad reduce)。六罪犯修复价值=loss/精度对齐+回归资产,应入 PR,但不等于"grad 差不多"达成。
- ⚠给 bayan:这是硬骨头,可能需要你醒后定夺——是继续钻 grad_norm(集成层),还是先按'RL 实效'
  路线(.3 用全修复 mlite 跑 15K 看 acc 是否健康涨点,grad_norm 差记为已定界 backlog)收口交付。

## 13:35 突破:grad 3.3x 主因收敛
- 12层 optimizer proxy(13503009)**首次 proxy 复现分叉**:reported grad_norm 12.67 vs MCore 2.75=4.6x
  (4层不够放大→之前 grad_abs=0 是假阴性,不是无罪);关 recompute(13503610)仍 12.67→排除 recompute。
- **源码嫌疑锁定 `run_microbatch_loop` 的 micro 累加/grad 归一口径**(verl↔engine 集成层,与 12:56 判断一致)。
- 有了可复现 12层 proxy(分钟级)+源码定位,收敛在望;worker weighted_logprob_unit 深挖中。

## 事实基线(睡前)
- loss 已对齐 0.01%;grad 3.3x 未闭环;罪犯 1-5 已修(明细见 GATE0.md/G2-RESULTS.md)。
- 15K megatron 锚点 step5 acc=0.7479;is/main 落后审计与 router primitive 修复方案在 .9 队列。
