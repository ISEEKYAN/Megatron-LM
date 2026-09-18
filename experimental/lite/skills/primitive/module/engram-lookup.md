# Engram Lookup Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.engram_lookup", kind="primitive", purpose="define and validate engram lookup",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def engram_lookup(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Uneven rows keep published FP8/scale bytes resident; group=None is local; every member participates even for empty requests; trainable sparse gradients return to owners; validate metadata before transport", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/engram_lookup.py",
        "api_and_state": "OwnerRowTransport, RowLookup, EngramTable, ShardedEngramTable, NgramHash, Engram; explicit row boundaries/process group/master policy",
        "details": read_source("megatron/lite/primitive/modules/engram_lookup.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use row-owned lookup; caller owns hash architecture and checkpoint names; no parameter/table offload", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/primitive/test_owner_row_transport.py and tests/unit/deepseek_v41/test_sharded_engram.py; uneven/empty ownership and backward", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
