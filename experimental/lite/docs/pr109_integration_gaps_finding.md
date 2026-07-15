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
