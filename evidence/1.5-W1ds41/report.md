参考源=`ISEEKYAN/Megatron-LM#242@edb328ca719056fd41927d86b6cf7be43a39d94d`，fetch 于 `2026-10-03T23:46:45Z` 前完成（随后立即记录 UTC）；验证基线=`01b252f37`；clamp 本地=`d392fa4b96fdc5acc0898f80f1fd455528b3b73a`。

# 1.5-W1-ds41：BLOCKED，停止于 master dtype 合同

仅验证分支 `validate/ds41-w4a8-on-227`。没有完成 DS4.1 W4A8 接线，没有宣称 CPU 模型或 GPU 0-diff。按任务第 2 条“不能就写清差异与对 0-diff 的影响，停下说明”停止；没有移植 primitive、改 optimizer 或改变默认执行路径。交付仅为本报告和可复现 CPU 阻塞探针；没有 PR、push、merge、GPU、任务 log、vk2/wiki 修改。

## 来源与前置读取

- 已读 llmrl2/CLAUDE.md、current_state、量化分层/双路径转换/DS4.1 合并历史 dead_ends，以及 W3/W2 report。
- 已从 delivery fetch #242 分支并核验 FETCH_HEAD 为上列 SHA。
- 远端 `feature/lite-w4a8-swiglu-clamp` fetch 返回 `couldn't find remote ref`；按任务指定回退至 `1.5-W1clamp` 本地 worktree，上列 SHA 是读取时的 HEAD。未改动其 worktree。
- 历史设计依据：`d3f45e9405c0c8fd029235b040697502510f457d` 引入的 `experimental/lite/docs/specs/deepseek_v41_optimizer_routing.md`，由 `git show` 读取；该文档不在本基线当前路径。其 Gradient and publication contract 明确要求 FP32 master/native FP32 GEMM wgrad，禁止先 BF16 舍入再转 FP32。

## 为什么不能原样保持 BF16 master

以下源码路径均相对 `experimental/lite/megatron/lite/`：

1. `model/deepseek_v41/lite/protocol.py:154` 只接受 optimizer=None 或 muon；`:204` 配置 native FP32 projection，`:208–212` 将全部 trainable 参数转 FP32 并设置 main_grad。不是只对某一层偶然 cast。
2. `primitive/modules/native_fp32_linear.py:16–18,25–31,84–91` 要求 FP32 master；forward 按 residual dtype 计算，backward 用 FP32 operands 做 wgrad，直接返回 FP32 leaf，避免 BF16 wgrad 中间值。
3. `model/deepseek_v41/lite/optimizer_groups.py` 将 routed expert w1/w2/w3 明确交给 HeadwiseMuon；`primitive/optimizers/headwise_muon.py:95–99` 拒绝 BF16 master，`:114–119` 拒绝非 FP32 gradient。MixedOptimizer 在 `:417–420` 再次拒绝任何非 FP32 master，`:480–481` 检查 native FP32 main_grad。
4. #242 的 W4A8 API 文档规定 BF16 master。其 backward 在 BF16 grad/x_hat 上执行 GEMM，再 cast 回 weight dtype；即使传 FP32 master 也不等价于 #227 原 native FP32 wgrad。底层 grouped API 没有强制拒绝 FP32 W，并不代表该组合已获合同或训练验证。

因此，“只跳过 protocol 的 FP32 cast”会在 native linear/Muon/mixed coordinator 处失败。跳过这些校验、对 routed experts另建 optimizer，或改变 gradient publication 都是新的训练数值合同，不能当成普通消费者接线悄悄带入。`optimizer=None` 仅能承载无更新前向，不能完成本任务的训练侧 master 合同。

## 对 0-diff 的具体影响与 CPU 证据

运行命令（仅 CPU，禁用可见 CUDA）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python evidence/1.5-W1ds41/check_master_contract.py
```

探针读取当前 production codec/native linear，原样提取 HeadwiseMuon `_validate_groups` AST 执行，以避免加载可选 NS backend。它不是完整 optimizer 构造或 step 测试。结果保存在 `master-contract.json`：

- 同一 validation method 接受 FP32 master、拒绝 BF16 master。
- BF16 residual + FP32 native linear 的 weight gradient 为 FP32；传 BF16 master 被 provider 拒绝。
- 同一个部署 `quantize_mxfp4`，输入 32 元素 block `[0.7501, 6, 0, ...]`：FP32 master 首 packed byte=114，先转 BF16 后首 byte=113，scale byte 均为127。反例没有更改 ties-down/zero-floor codec，差异只来自量化输入舍入。

W4A8 前向在 W 侧只依赖量化后的 codes/scales，**FP32 master 本身并不必然阻止同代 actor↔rollout 0-diff**。若两端都量化同一 FP32 master，可在相同 packed bytes 上验证前向；但这偏离 #242 BF16 master 合同，且反向合同也待定。若训练临时 cast BF16，而 native resync exporter 仍读取 live FP32 master，会出现上述实际字节差异。若决定导出也 cast BF16，必须同步 load↔export 及 checkpoint/resync spec，不能只改训练入口或藏在 exporter 中。

本次没有改任何产品文件，默认实现源码保持 01b252f37；**没有执行默认模型输出逐字节回归**。未执行 DS4.1 tiny 完整 forward↔W2 独立参考、fake-quant↔native resync/load-export 镜像测试，也未运行 optimizer 两步；它们全部仍是未通过的验收项，不以 leaf probe 替代。

证据脚本按仓库版本 isort 5.13.2 / black 24.4.2 格式化并检查；产品目录与基线 `git diff --exit-code` 通过，`git diff --check` 通过。pre-commit wrapper 尝试在临时目录建环境时因 PyPI DNS 解析失败退出，故改用已有 pinned formatter 直接执行；不宣称 wrapper 全绿。该 evidence 路径本身不匹配 lite hooks 的 `^experimental/lite/.*\.py$` 范围。

## 需要 bayan 裁定后才能恢复

请裁定 routed expert 的 master/gradient 合同：保留 #227 FP32 master 并明确允许偏离 #242（两端都直接量化相同 FP32 输入）；或要求 BF16 master并授权调整该路径的 optimizer/native gradient 合同。另一种 FP32 optimizer master + BF16 量化视图仍是 FP32 master，不能冒称 BF16 master；采用它也须明确两端一致的 view 和镜像语义。#227 方案 B/例外的最终路线继续由 bayan 决定，本验证不迁移实现到任何 PR 分支。

裁定后再 cherry-pick #242 及 clamp 的必要提交，记录全部来源 SHA，保留 opt-in/EP1；接 A8 g128 动态、部署 W codec g32、config.swiglu_limit、原 top-k 槽位 FP32 FMA。先完成 tiny 模型/W2 独立 CPU 参考、字节镜像和默认基线回归，再提交验证实现。

## 后续 8 卡 GB200 gate（计划，未执行）

先通过无 GPU 的完整 CONFIG_ONLY init，核对 dtype、量化输入、backend、EP1、exporter、版本；完成 CPU gates。需先恢复 W2 指定的可跨 allocation 复用环境，锁定 W2 vLLM `5db5732a7d988a755ad5376813d1847455e40c16`、DeepGEMM `dfad2230d33ac6bb853b00e56a1c94664f4fe195` 和最终训练 SHA。W2 当前 guard 只支持 TP/EP/DP/PCP/SP=1，不能将 8 卡数量直接解释为获支持的 TP8 rollout；先用单卡可容纳 tiny 配方，在 8 卡上跑独立进程的相同 snapshot/分块扰动。全尺寸、多进程 DS4.1 需另解并行支持闸。

每个进程 EP1，先局部 A1/A2 codes/scales、clamp10、两次 GEMM、FP32 top-k 输出逐位检查；再两次真实 optimizer step→导出→resync，与同代 cold load 比较 W/scale/padding 及 logits/token logprob 的严格零差。覆盖 M=1/31/32/33/128/129/513、空 expert、伙伴/重排/chunk、eager 后 graph，保存 snapshot hash、routes/FP32 weights、中间 bytes 和每代 staging/缓存地址证据。CPU GEMM 不等价于 tensor-core 规约；CPU通过不能替代 GPU。按 HSG 2节点×4卡 interactive、W&B、RUNNING五分钟内首诊及卡住取消留账的既定纪律执行。本任务未申请资源。
