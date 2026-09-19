# Bound Training Primitives

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.bound_training", kind="primitive",
    purpose="route bound HF storage, row memory, paired streams, vision, block codecs and owned optimizer/DDP",
    imports=["basic.constitution"], calls=["primitive.validate"],
    inputs=["task", "files", "reference", "budget"],
    outputs=["routes", "validation", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
# Paths are relative to megatron/lite/primitive and tests/unit, respectively.
ROUTES = {
    "config_fields.py": "deepseek_v41/test_redo_parity.py",
    "ckpt/binding_records.py": "deepseek_v41/test_redo_parity.py",
    "modules/engram_lookup.py": "deepseek_v41/test_redo_engram_residency.py",
    "modules/owner_row_transport.py": "deepseek_v41/test_redo_engram_residency.py",
    "modules/row_memory_build.py": "deepseek_v41/test_redo_engram_residency.py",
    "modules/paired_stream.py": "deepseek_v41/test_redo_parity.py",
    "modules/image_data.py": "deepseek_v41/test_redo_parity.py",
    "modules/vision.py": "deepseek_v41/test_redo_parity.py",
    "modules/vision_training.py": "deepseek_v41/test_redo_parity.py",
    "modules/native_fp32_linear.py": "deepseek_v41/test_redo_codecs.py",
    "quantization/mxfp8.py": "deepseek_v41/test_redo_codecs.py",
    "quantization/nvfp4.py": "deepseek_v41/test_redo_codecs.py",
    "optimizers/headwise_muon.py": "deepseek_v41/test_redo_parity.py",
    "optimizers/owned_groups.py": "deepseek_v41/test_redo_parity.py",
    "optimizers/sinkhorn.py": "deepseek_v41/test_redo_parity.py",
    "optimizers/staged_update.py": "deepseek_v41/test_redo_parity.py",
    "parallel/owned_ddp.py": "deepseek_v41/test_redo_parity.py",
}


def bound_training(task, files, reference, budget):
    routes = select_matching_paths(files, ROUTES)
    if not routes:
        return out_of_scope("no bound training primitive touched")
    if reference is None:
        return blocked("require independent numerical or ownership reference")
    invariants = [
        "HF save/export reject unsupported resync symmetrically; archived bytes survive reload",
        "row owners retain storage; uint8 transport preserves FP8 scale bytes",
        "paired PP output and gradients equal the unpartitioned execution",
        "image tokens execute vision/aligner with finite nonzero gradients and updates",
        "default quantized execution uses real codecs, bounded error and exact replay checks",
        "optimizer commits all owners or none; dense/expert/row reductions use their actual groups",
    ]
    validation = []
    for route in routes[:budget.max_candidates]:
        result = primitive.validate(task, primitive=route, implementation=load_source_and_tests(route, reference, invariants), budget=budget)
        validation.append(result)
        if not result.done:
            return blocked("primitive validation failed", evidence=validation)
    if len(validation) != len(routes):
        return blocked("validation budget did not cover every touched route", evidence=validation)
    smoke = require_real_gpu_smoke(task, cases=["multimodal", "quantized", "ep2", "cp2"], skip_is_failure=True)
    if not smoke.done:
        return blocked("GPU composition smoke failed", evidence=smoke)
    return done(routes=routes, validation=validation, risks=["unit tests alone do not certify engine e2e"])
```
