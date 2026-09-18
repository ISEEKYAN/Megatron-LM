# Paired Payload Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.paired_payload", kind="primitive", purpose="define and validate paired payload",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def paired_payload(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "h is floating [B,S,HC,D], p is floating [B,S,HC]; optional ced pair travels together; published bytes/index metadata cannot require gradients; selections/positions integer; fixed tuple field order", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/paired_payload.py",
        "api_and_state": "PairedPayload, PAYLOAD_FIELDS, tensors, from_tensors, differentiable",
        "details": read_source("megatron/lite/primitive/modules/paired_payload.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use paired residual pipeline carriers; caller owns layer cuts, attention state construction and scheduling", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/primitive/test_paired_payload.py; tests/unit/deepseek_v41/test_pipeline.py and test_pp_stages.py; payload roundtrip, gradients and stage boundaries", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
