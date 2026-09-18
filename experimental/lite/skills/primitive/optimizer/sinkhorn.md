# Sinkhorn Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.optimizer.sinkhorn", kind="primitive", purpose="define and validate sinkhorn",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def sinkhorn(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Fresh Nesterov direction; K=11, tau=1e-3, eps=1e-20, gamma=0.18; no warm-start state; global norm/mask domains follow explicit groups; atomic nonfinite skip", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/optimizers/sinkhorn.py",
        "api_and_state": "sinkhorn_direction, Sinkhorn; row_group/column_group and staged update",
        "details": read_source("megatron/lite/primitive/optimizers/sinkhorn.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use logical parameter matrices including explicit row sharding; not a replacement for model owner/replica reduction contracts", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/primitive/test_sinkhorn_algorithm.py and tests/unit/deepseek_v41/test_sharded_engram.py; independent scalar arithmetic, momentum and resume", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
