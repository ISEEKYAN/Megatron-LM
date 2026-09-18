# Ds41 Index Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.quantization.ds41_index", kind="primitive", purpose="define and validate ds41 index",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def ds41_index(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Post-rotary E2M1 group32 with E8M0 scale; clamp amax before scale; preserve nibble packing/ties; switch independent from main KV", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/quantization/ds41_index.py",
        "api_and_state": "quantize_index, fake_quant_index; enabled switch",
        "details": read_source("megatron/lite/primitive/quantization/ds41_index.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use index Q/K codec; not group16/E4M3 main-KV quantization", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/deepseek_v41/test_quantization.py; independent codec bytes, STE and disabled path", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
