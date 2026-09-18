# Vision Training Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.vision_training", kind="primitive", purpose="define and validate vision training",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def vision_training(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Owner exposes vision/norm/aligner/image vectors/embed/mask; weight copy precedes forward; language backward precedes vision backward; mask/restore only at idle; abort releases pending graphs", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/vision_training.py",
        "api_and_state": "VisionTrainability, VisionSchedule; encoder/norm/aligner/delimiter booleans and explicit device",
        "details": read_source("megatron/lite/primitive/modules/vision_training.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use serial external vision execution with owner parameters; no overlapping microbatches or model-specific owner-key mapping", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/primitive/test_vision_training.py and tests/unit/deepseek_v41/test_model.py; model composition exercises frozen/trainable owners and restart", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
