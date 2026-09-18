# Image Data Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.image_data", kind="primitive", purpose="define and validate image data",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def image_data(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Preserve resize token budget and image order; spans must be ordered/disjoint/in bounds with equal row widths; expand delimiter FP32 owners before residual cast", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/image_data.py",
        "api_and_state": "ImageConfig, ImageInput, plan_image_grid, preprocess_image, prepare_image_inputs, merge_image_embeddings",
        "details": read_source("megatron/lite/primitive/modules/image_data.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use decoded PIL images and tokenized prompts; caller owns URL loading/tokenizer and subsequent residual expansion", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/primitive/test_vision.py; pinned pixels/grid and multi-image delimiter gradients", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
