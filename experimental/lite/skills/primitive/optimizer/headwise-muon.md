# Headwise Muon Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.optimizer.headwise_muon", kind="primitive", purpose="define and validate headwise muon",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def headwise_muon(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Explicit logical heads remain independent; matrix layout validated; stage/prepare precedes atomic commit; nonfinite candidates do not partially publish; restore preserves owner state", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/optimizers/headwise_muon.py",
        "api_and_state": "StagedMatrixOptimizer, HeadwiseMuon, MixedOptimizer; logical matrix_shape/partitions and FP32 owners",
        "details": read_source("megatron/lite/primitive/optimizers/headwise_muon.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use declared logical matrices and explicit backend policy; model owns parameter inventory and role selection", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/deepseek_v41/test_headwise_muon.py and test_model.py; independent head updates, staged skip and restart", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
