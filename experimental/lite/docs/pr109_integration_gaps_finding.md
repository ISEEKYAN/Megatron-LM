# PR#109 (feat/ds4-grpo-debridge-verl-native-fp8) 集成缺口 finding

**背景**：托管夜跑，尝试用 PR#109 集成分支跑 8卡 smoke→128 hero。PR#109 疑似**从未端到端跑过**——发现 2 处集成缺口。第 1 处已修+commit，第 2 处是架构冲突，**留 bayan 醒来裁**。

## 缺口 1（已修，commit `8bd14313b` 已 push 分支）
fp8 debridge 的 **config→engine→checkpoint resync 管道半 merge**。fp8 commit `0cba56d7a` 加了叶子（`resync.py::export_resync_weights`、`checkpoint.py` target='vllm_checkpoint' 路径）+ run 脚本设 `engine.resync_format=vllm_checkpoint`，但漏了 4 块管道：
- `verl_mlite/engine/config.py`：`resync_format`/`resync_config` 字段 + 校验
- `verl_mlite/config/engine/mlite.yaml`：`resync_format: null` + `resync_config: {}`（Hydra struct 声明——不加则 `Key 'resync_format' is not in struct`）
- `megatron/lite/runtime/contracts/weights.py`：`ResyncFormat` enum（新文件）
- `mlite_engine.get_per_tensor_param`：`resync_format→export target/resync_config` 转发

已从 validated debridge overlay port 齐 4 块并 push（纯英文 commit）。

## 缺口 2（架构冲突，**未动，待 bayan 裁**）
`verl_mlite/engine/mlite_engine.py:28,39`：
```python
from verl_mlite.compat import _patch_bucketed_weight_sender, load_verl_engine_api
...
_patch_bucketed_weight_sender()   # module-load 时调用
```
但 **PR#109 的 compat.py（fp8 commit 重写 862 行）没有 `_patch_bucketed_weight_sender`**（`load_verl_engine_api`/`apply_runtime_patches` 都在）。→ 实例化 config 时 import 崩：
```
ImportError: cannot import name '_patch_bucketed_weight_sender' from 'verl_mlite.compat'
```

**为何不能简单 port**：debridge 的 `_patch_bucketed_weight_sender` 牵一整簇 compat 内部——`_patch_bucketed_weight_transfer` / `_weight_sync_probe_enabled` / `_BUCKETED_SENDER_MODULE` / `_instrument_bucketed_weight_sender` / meta_path finder。这是 **fp8 resync IPC 字节对齐（Fix-A）** 的基础设施（防 fp8 8-byte tensor 非对齐 offset → 接收端 `.view` 崩，即最初 128 卡 resync 崩的根因域）。

**架构冲突点**：托管操作员曾把 rollout/probe 相关代码当"残留"清理，可能**误把 `_patch_bucketed_weight_sender`（Fix-A 载重，正确性需要）当残留删了**。这是 **bayan curate 意图 vs fp8 正确性需要** 的冲突，属架构决定，不瞎改。

**待 bayan 裁的 2 选 1**：
- **restore**：把 bucketed-weight-sender compat 簇从 debridge 恢复进 PR#109 compat.py（fp8 resync 字节对齐需要，我判断**需要**）
- **删 import**：若确认 PR#109 的 fp8 路不经这条字节对齐（用别的机制），则删 `mlite_engine.py:28,39` 的 import+调用

## 今晚托管走向（路 B）
PR#109 缺口 2 未解 → 今晚 smoke/hero **改跑 validated debridge overlay**（`source-e419c9768-debridge`，已端到端证过跑通 128 gibberish fp8，全套 plumbing 在），拿到 bayan 要的 "print response + 128 乱码判" 结果。PR#109 分支 resync plumbing 修（缺口1）保留，compat 簇冲突（缺口2）留 bayan 醒来裁。

## 附:重复loop廉价复验(item1,留bayan裁)
- **假设**:128 full val 的重复loop("We will think step-by..."/"We will be stepwise:")=greedy(temp=0)解码退化,非fp8 resync腐蚀(输出是连贯真英文非乱码,已证resync数值对)。
- **廉价验证(单炮,成本铁律)**:job `14046246`(8卡 full DeepSeek-V4-Flash + `VAL_TEMPERATURE=1.0`+`VAL_DO_SAMPLE=True`+gmu0.50,submit `submit_ds4_8card_full_valtemp_loopcheck.sh`)。**发时 PENDING 排队**(高优先级作业占队),未撤不retry,待节点自然跑。
- **判读(job跑完看val_generations/0.jsonl)**:temp=1.0后**重复消失/多样化**=greedy退化实锤(loop非resync,拆桥结论更硬);**仍重复**或**actor OOM(full模型8卡PP1EP8CP1装不下)**=留bayan定(换instruct-format/rollout-only load/或128重验)。
- item1本就bayan裁,此复验仅为强化结论,不阻塞主结论(拆桥数值真通已由4层证据+128连贯英文闭环)。

## ★loop复验结果(job14046246跑完)=假设证伪,结论修正★
temp=1.0采样(full模型)输出=**随机多语言垃圾token**(`nakalista窒息的 hilab`/`EDMF accountingKaginharian`),**不是**变成"多样连贯文本"。
- temp=0 greedy=连贯英文但重复("We will think step-by");temp=1.0=垃圾。
- **"loop=greedy退化temp能修"假设❌证伪**:temp>0不是修loop而是变垃圾。
- **诚实解读**:greedy top-1连贯英文=argmax合理(vs减层greedy就垃圾,是正向);但分布尾部噪声(temp>0采到垃圾)=base/未tune模型高熵 **OR** resync轻微数值不精确影响logit尾部,**这层证据分不开**。
- **结论修正(不夸大)**:前3层CPU/静态证据仍硬(resync结构+逐块数值正确);greedy连贯=正向;**但生成分布质量存疑(base vs resync尾部)真留bayan深究item1**(需换instruct模型/对比bf16 baseline logits才能定)。**对外宜说"数值正确性证据强+greedy top-1连贯,生成分布质量待验",不宜声称"完全真通"。**
- job崩仍val后packing 206>128(不挡val文本)。
