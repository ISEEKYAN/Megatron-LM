# Router Replay Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.router_replay", kind="primitive", purpose="define and validate router replay",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def router_replay(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Replay substitutes expert indices but gathers live scores; preserve packing/mask order and forward/backward lifetime; counters distinguish payload presence from actual substitution", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/router_replay.py",
        "api_and_state": "RouterReplay, RouterReplayAction, build_r3_replay_mask; record/forward/backward actions",
        "details": read_source("megatron/lite/primitive/modules/router_replay.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use recorded routes with matching shape and schedule; changed==0 remains diagnostic rather than a fabricated correctness failure", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/primitive/test_router_replay_evidence_unit.py and tests/unit/deepseek_v41/test_router_replay.py; forced rank swap plus backward", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
