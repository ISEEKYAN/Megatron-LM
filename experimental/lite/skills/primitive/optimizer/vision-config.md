# Vision Config Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.optimizer.vision_config", kind="primitive", purpose="define and validate vision config",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def vision_config(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Immutable policy records; visual numeric values finite and nonnegative; no role, owner or release-key dispatch in this module", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/optimizers/vision_config.py",
        "api_and_state": "VisionOptimizerConfig, OptimizerConfig; lr/ns_steps/coefficient_type/clip_grad/vision_policy",
        "details": read_source("megatron/lite/primitive/optimizers/vision_config.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use explicit caller-supplied optimizer policy; model composition retains role/head/owner mapping and algorithm selection", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/primitive/test_vision_config.py; tests/unit/deepseek_v41/test_optimizer.py and test_sharded_engram.py; serialization and policy composition", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
