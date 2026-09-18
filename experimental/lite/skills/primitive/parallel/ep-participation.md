# Ep Participation Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.parallel.ep_participation", kind="primitive", purpose="define and validate ep participation",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def ep_participation(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Bounded host store rendezvous; calls serialized in identical rank order; failed groups must be torn down; host participation does not prove CUDA completion", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/ep_participation.py",
        "api_and_state": "check_ep_participation(group, phase)",
        "details": read_source("megatron/lite/primitive/modules/ep_participation.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use before native EP collectives; do not treat as collective retry or CUDA synchronization", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/deepseek_v41/test_expert_parallel.py::test_native_ep_missing_peer_fails_before_transport", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
