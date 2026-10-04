参考源=`ISEEKYAN/Megatron-LM#242@edb328ca719056fd41927d86b6cf7be43a39d94d`，fetch 于 `2026-10-03T23:46:45Z` 前完成；clamp 本地=`d392fa4b96fdc5acc0898f80f1fd455528b3b73a`（该远端分支当时未发布）。

# 1.5-W1-ds41：FP32 master 扩展与 CPU 验证载体

秘书已裁定保留 #227 的 FP32 master/native FP32 wgrad 合同。本报告取代 initial-blocker.md 的阻塞结论。实现只在 `validate/ds41-w4a8-on-227`，基线 `01b252f37`；未 push、未开 PR、未合并 PR、未申请/访问 GPU、未写任务 log、未改 vk2/wiki。不是 GPU 或完整原生混合精度 actor↔rollout 0-diff 验收。

## 移植来源

以下均用 `cherry-pick -x`，保留原作者和来源 SHA：

| 来源提交 | 验证分支提交 | 内容 |
|---|---|---|
| 607fb1eb3bf6756da1b0727b13a70701695f54fb | 33de5b8fc | W4A8 primitive / Qwen 消费者 |
| 44a0d386a9e02f5071cb41be54b1529d83f64573 | ee650a04d | DeepGEMM scales / shape guards |
| 51c704c4cc8adeb38dba7f6ce6006b8f767ac134 | cac99ec78 | FP32 top-k FMA combine |
| edb328ca719056fd41927d86b6cf7be43a39d94d | 64dfbebfc | EP1 / unweighted expert rows |
| d392fa4b96fdc5acc0898f80f1fd455528b3b73a | 7bb32108b | SwiGLU clamp |

W2 独立参考来自 vLLM `5db5732a7d988a755ad5376813d1847455e40c16` 的 `tests/kernels/moe/w4a8_reference.py`，原样复制至 DS4.1 测试目录，保留 Apache-2.0 头；SHA256=`c4732684ce30d226dfce43c71dc9c2ea091bfbd248a64b03bf7573b812f279e5`。未使用训练 quantizer/dispatch/combine 构造预期结果。

#231 resync transport 参考为 `055e4112d839881710e5e7e9523ea2d8eccb578a` 的 `model/deepseek_v41/lite/resync.py`，证据目录保存原样副本 `resync_231_reference.py`；SHA256=`a07f12c9dc8672641bd9c98c6fbaca8d3594f9d4cfd6b096d4bdd705418e2ac7`。从本地 1.2-impl 读取后与指定 commit 的 git blob hash 对拍一致。

## 实现与使用合同

- `primitive/quantization/w4a8_experts.py` 明确支持 BF16/FP32 master，拒绝其他 master dtype。前向直接把 live master 交给部署 `quantize_mxfp4`，无预先 BF16 cast；仍为 g32、ties-down、zero floor。
- FP32 master 的 STE wgrad：`grad.float().T @ x_hat.float()`，局部禁用 autocast，直接返回 FP32 到 leaf。BF16 master 保留原 backward。输入梯度仍按 BF16 activation 合同。**这是超出原 #242 BF16-master 消费者合同的扩展，后续须回补 primitive PR**。
- DS4.1 `ImplConfig.w4a8_experts=False` 默认不变。开启后要求 BF16 residual、EP1、无 DeepEP、unfused dispatch、hidden/intermediate K 可被128整除。配置入口、直接构造的 MoE 和 primitive 各自检查所属条件。
- 仅 routed experts 走新 primitive；FC1 按行拼 w1/gate+w3/up，FC2 读 w2。没有替换叶子 Parameter，没有改变 optimizer/ckpt binding；拼接保持 FP32 数值及每个 g32 block 原样。A1/A2 动态 E4M3 g128，SwiGLU 取配置 limit（此代理为10），FC2 后按原 top-k slot 顺序 FP32 FMA，最后舍入一次 BF16。
- HeadwiseMuon/MixedOptimizer、native_fp32_linear、checkpoint encoder/loader 均未改。shared experts、router、attention、mHC 产品逻辑未改。
- 用法见 `experimental/lite/examples/verl/DS41_W4A8_VALIDATION.md`。生产选择 `quantized=True,w4a8_experts=True,optimizer='muon',dtype=bfloat16`，显式给 OptimizerConfig；CPU诊断 `quantized=False,w4a8_experts=True` 只把非专家投影切到既有浮点诊断路径。

## CPU 验证结果

复跑入口：`bash evidence/1.5-W1ds41/run_cpu.sh`。脚本固定无可见 CUDA、单线程，保存 pytest 日志和逐字节回归 JSON。环境为 Torch `2.12.1+cpu`，Python3.13；Core `/tmp/mrg@1c4df34a5c53ed43d3fde36873ad5c4a671ee884`，Emerging Optimizers `/tmp/ds41-f12-emerging-audit@b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16`（0.3.0），缺失 absl-py 2.3.1 隔离安装在 `/tmp/w1ds41-deps`。路径可由脚本 W1DS41_* 环境变量替换；未 mock Muon backend。

1. 新增 DS4.1 suite：**6 passed**，见 ds41-cpu.txt。真实 `build_model` tiny 模型：40层原 DS4.1 topology，hidden=128、expert intermediate=128、2专家/topk2、shared expert保留、关闭 Engram；3 token，BF16 residual/FP32 master。三代 snapshot，含两次实际 MixedOptimizer/Muon 更新，量化 payload 每次确实改变。
2. 每代直接捕获训练 forward 真正调用 quantizer 的 packed W/scales，与实际 `checkpoint.export_checkpoint` 逐字节对拍；每代240个专家投影，共720对 W/scale。包含 `.7501` 中点邻域、正负零、tiny、零 block。导出/前向不修改 master。#231 transport encode→decode 后这些 bytes 仍一致。
3. 同代导出 W/scale 用独立 MXFP4 nibble/UE8M0 decoder 解码，再用 **W2 原始 A8/clamp/FMA 参考**、独立按 routes 分派和 CPU FP32 matmul 计算专家输出。将 tiny 模型全部 routed MoE 换成该导出参考后，三代 full-model logits 严格逐位相等。非专家部分共用同一模型实现/同代 master；这不是全 DS4.1 vLLM 模型的独立参考。
4. 最后一代实际 `save_model→load_model→export_checkpoint`：恢复的每个专家 FP32 master 与原 master 字节相等，W/scale再次导出相等（含 `mlite_masters` 精确恢复，未对已量化权重二次量化冒充 master）。archival payload 为合法测试容器中的合成 bytes，不执行视觉/DSpark。
5. 原生 FP32 wgrad 单独对照 FP32 GEMM，刻意令 backward 在 CPU BF16 autocast 内执行；梯度逐位相等，且确实不同于 BF16 舍入后再 widen 的值。整模两步后所有 routed leaf gradients 为 FP32，optimizer 原检查通过。
6. fail-closed：FP32 residual、DeepEP、EP>1、fused permute、K不整除均拒绝。扩展回归 **81 passed / 9 GPU skipped**，见 regression-cpu.txt；包含 primitive、Qwen3消费端、DS4.1 bindings/parallel guards。GPU skips 不计数值通过。
7. 从 `git archive 01b252f37` 独立解出 baseline，用同一测试脚本、相同Core/backend/种子、独立进程执行，断言实际导入来自各自 tree。**2153个 tensor、53,781,316 bytes 全相等**，含 full tiny FP32 residual logits/loss/参数/梯度，以及默认 MoE 的 BF16 forward/backward；不使用 mHC adapter。结果见 default-parity.json，聚合 SHA256=`cdc72275cffb999b30af0decd005eef36cf98b3a3bbe928efa9f5844bd02330c`。此项验证默认分支行为，不声称未运行的默认 quantized GPU 全模型回归。

## 必须保留的边界与待办

- **格式 caveat**：已运行仓库 pinned isort5.13.2/black24.4.2 hooks 两轮，组合仍因多行import互改失败；Black独立检查通过、`git diff --check`通过。精确改写见 `isort-diff.txt`，基线同类失败见 `baseline-isort-diff.txt`，hook输出见 `pre-commit.txt` / `black-check.txt`。例如 `protocol.py:13` 原有括号形式被isort改为反斜线形式，Black再还原；`moe.py:177` 的多符号导入也有单行/逐行冲突。这是已记录于 `isort_black_pingpong_never_converges.md` 的仓库配置问题，本验证分支保留Black格式，没有改全局格式配置或宣称hooks全绿；后续PR迁移须解决格式闸。
- FC1 的 FP32 行拼接会分配临时buffer，未做GPU显存/吞吐profile；本次只交数值验证载体，不承诺性能持平。
- **CPU BF16 full model 的既有 Core mHC 限制**：native_h_post_bda 的 bmm 不接受 FP32 mix coefficients×BF16 residual。W4A8 整模测试只对该 CPU provider 使用其原有 native 算术的 FP32 operands适配，两个比较臂相同，调用者仍保留 BF16输出边界。未改产品 mHC，也没有替换 expert/quantization/optimizer。这项 harness 限制必须在 GPU原生 mHC 路径补验；默认基线测试用现有可运行的 FP32 residual全模另加 BF16 MoE，不使用适配。
- **#227 尚没有 #231 在线 resync 入口**：当前 protocol 对 target/resync_config 仍拒绝。此处验证的是本分支实际 live-master HF exporter + 指定 #231 transport 字节合同，没有声称已贯通 VERL IPC/native receiver。按 #227 最终路线迁移时复用获准的 #231 resync 交付，不在此分支重建 exporter/receiver。
- CPU GEMM不是 tensor-core归约模拟器，CPU exp不是 GPU exp 的逐位证明；全部 GPU测试仍未运行。不能从本报告推出 DeepGEMM、batch invariance、完整 rollout logits/token logprob 或图模式 0-diff。
- #227方案B/例外由 bayan 裁定；本地实现和 primitive FP32扩展随最终路线迁移，不推任何 PR分支。

## 下一步 8×GB200 严格零差方案（尚未执行）

1. 无GPU CONFIG_ONLY 穿完整 actor/rollout init：确认本开关、FP32 master、BF16 residual、同一量化输入、clamp10、g128/g32、EP1、禁融合dispatch、#231 custom resync入口与generic QAT exporter不混用。先集成获准resync入口，冻结最终训练SHA。
2. 恢复 W2可跨allocation复用的持久环境，锁定 vLLM `5db5732a7d988a755ad5376813d1847455e40c16` 和 DeepGEMM `dfad2230d33ac6bb853b00e56a1c94664f4fe195`；环境/编译产物hash、实际API及backend均核实后运行。
3. W2当前guard只支持TP/EP/DP/PCP/SP=1，**不能直接将8卡解作TP8 rollout**。先2节点×4卡运行8个独立EP1 tiny代理，覆盖种子、batch/chunk扰动；全尺寸多进程支持另立闸。每进程两次实际optimizer step，每代训练量化bytes→export→receiver/native repack严格相等，再比较同代cold logits/token logprob。
4. 先A1/A2 codes+scale、clamp、两次GEMM、FP32 top-k，后整模；覆盖M=1/31/32/33/128/129/513、空专家、重排/伙伴/chunks1/7/32/64；eager严格零差后才测CUDA graph。记录master/snapshot hash、routes、FP32 weights、中间bytes、缓存地址、padding、每代staging归零。不能用 BF16 MXFP4 backend替代指定 FP8×MXFP4 DeepGEMM。
5. 后续申请遵守HSG入口、≤2节点interactive、W&B entity=megatron-core-moe-dev、RUNNING五分钟内首诊、卡住取消留账。此次无GPU allocation；剩余GPU验收不冒充本地完成。
