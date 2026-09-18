# Vision Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.vision", kind="primitive", purpose="define and validate vision",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def vision(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Height then width rotary phases; positive patch grid; head dimension divisible by four; FP32 norm owners; differentiable spatial padding and alignment", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/vision.py",
        "api_and_state": "ViT, Aligner; vision_patch_size, vision_dim, vision_n_heads, vision_inter_dim, vision_n_layers, vision_rope_theta, vision_downsample_ratio, dim",
        "details": read_source("megatron/lite/primitive/modules/vision.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use a patch ViT and spatial aligner; no image decoding, freezing policy or model checkpoint mapping", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/primitive/test_vision.py; compare pinned forward and parameter/input VJP in FP32/BF16", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
