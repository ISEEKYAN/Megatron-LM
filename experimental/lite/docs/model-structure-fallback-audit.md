# MLite model-structure fallback audit

MLite model implementations own their structural contract. Missing layers, stage modules, or
configuration fields are programming/configuration errors and must not be converted into an empty
collection, `None`, or a feature-disabled default.

## Gate

`experimental/lite/tools/check_model_structure_fallbacks.py` parses production Python files across
`experimental/lite` (tests are excluded). Inside `megatron/lite/model`, every `getattr` call with a
default value and every `hasattr` call is guarded. In other production directories, the same rule
guards registered MLite-owned structure fields such as `layers`, `mtp`, `layer_indices`, and
`sp_params`. A guarded call fails with `MLITE001` unless its stable signature is registered in
`model_structure_fallback_allowlist.json`. The allowlist requires a specific reason, enforces the
number of matching call sites, and rejects stale entries.

The rule is intentionally conservative: model code is denied by default, while real boundaries to
PyTorch, Transformers/Hugging Face, distributed wrappers, heterogeneous LoRA adapters, and generic
benchmark inputs are registered individually. The checked-in allowlist contains 29 such calls (28
stable signatures; one generic export signature occurs twice) and is the per-call inventory for
retained boundary fallbacks.

The check runs as both a local pre-commit hook and the `MLite model structure lint` GitHub Actions
workflow.

## Existing-code disposition

The audit examined 94 guarded call sites: 65 MLite-owned structural fallbacks were removed and 29
interface-boundary calls were retained in the explicit allowlist.

| Call sites | Classification | Resolution |
| --- | --- | --- |
| `deepseek_v4/lite/protocol.py`: `chunk.model`, `model.layers`, `model.mtp` | Owned DS4 chunk structure; this was the activation-checkpoint silent no-op | Read `chunk.layers` and `chunk.mtp` directly; shared recompute/offload primitives now reject a non-empty policy with zero layers |
| DS4, GLM-5, Kimi K2, and Qwen3.5 checkpoint modules: `layer_indices`, `embed`, `mtp_embed`, `norm`, `head`, `mtp` | Owned pipeline-stage structure; optional stages are represented by declared fields set to `None` | Use direct fields and preserve the existing `is not None` stage checks |
| All five model families: `dispatcher._local_tpe_list` | Owned dispatcher field initialized by every MLite dispatcher implementation | Use direct access |
| All five protocol `build_model_config` functions: `hasattr(cfg, override)` | Owned dataclass plus user-supplied override; unknown keys were silently ignored | Validate against `__dataclass_fields__` and raise `ValueError` for an unknown override |
| GLM-5 and Kimi K2 protocols: `num_nextn_predict_layers` | Required field of the owned model config | Use direct assignment when MTP is disabled |
| Qwen3-MoE, Qwen3.5, GLM-5, and Kimi K2 protocol `vocab_size` helpers | Required field of the protocol-owned config | Use direct access instead of nested/default fallback |
| Qwen3.5, GLM-5, and Kimi K2 constructors: `train_config.recompute_modules` and `train_config.deterministic` | Required internal construction contract | Use direct access so incomplete construction fails immediately |
| GLM-5 router/model: `aux_loss_alpha`, `tie_word_embeddings` | Owned config fields that were implicit because upstream HF config omits them | Declare the fields with the same defaults in `Glm5Config`, then use direct access |
| Qwen3.5 model: `config.mrope_section` | Declared owned config field | Use direct access while retaining `None` as its explicit value |
| DS4 forward adapter: `batch.position_ids` | Declared `PackedBatch` field | Use direct access; `None` remains a valid explicit value |
| Shared protocol helper: `model.cross_entropy_fusion` | Field installed on every chunk by the owning protocol before forward | Use direct access |
| Qwen3.5, GLM-5, and Kimi K2 recompute module maps | Owned layer fields whose optionality is represented by explicit `None` | Read the layer field directly; dynamic selection uses non-defaulted `getattr`, so a miss raises |
| Native LoRA alpha inference: `rank`, `scale` | Required fields on all native LoRA module variants yielded by the iterator | Use direct access |
| FSDP2/dist-opt optimizer helpers: `model.sp_params` | Required native-model optimizer contract; DS4 was the only family not declaring it | Declare an empty DS4 `sp_params` list (DS4 has no TP/SP) and use direct access in shared optimizers |
| dist-opt setup: `engine_cfg.deterministic` | Required field on every native implementation config | Use direct access |

The 29 retained calls and their individual classifications live beside the rule in the allowlist;
adding another boundary requires an exact signature and a reviewable reason.

## Commands

```bash
python experimental/lite/tools/check_model_structure_fallbacks.py
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" pytest \
  experimental/lite/tests/unit/model/test_recompute_fail_loud.py
```
