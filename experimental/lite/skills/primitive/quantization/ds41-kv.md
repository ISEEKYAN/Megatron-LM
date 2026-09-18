# Ds41 Kv Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.quantization.ds41_kv", kind="primitive", purpose="define and validate ds41 kv",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def ds41_kv(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Post-rotary E2M1 group16/E4M3; official round-to-nearest ties; detached-scale identity STE only in supported phase; preserve byte publication and finite validation", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/quantization/ds41_kv.py",
        "api_and_state": "QuantizedValues, quantize_main_kv, fake_quant_main_kv; enabled and phase",
        "details": read_source("megatron/lite/primitive/quantization/ds41_kv.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use main-KV activation codec; not weight codec or index group32 policy", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/deepseek_v41/test_quantization.py; independent scale/nibble reference and phase/input rejection", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
