# Hyper Connection Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.hyper_connection", kind="primitive", purpose="define and validate hyper connection",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def hyper_connection(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Hidden layout [B,S,HC,D]; pre-mix [B,S,HC]; FP32 weighted contraction; shifted pre/post orientation preserved", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/hyper_connection.py",
        "api_and_state": "expand_hc, contract_hc, mix_residual, HCMixes, RMSNorm",
        "details": read_source("megatron/lite/primitive/modules/hyper_connection.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use paired shifted residual streams; model owns layer wiring and residual source selection", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/deepseek_v41/test_layers.py and tests/unit/primitive/test_vision.py; independent orientation and accumulated delimiter VJP", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
