# Csa2 Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.csa2", kind="primitive", purpose="define and validate csa2",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def csa2(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Preserve compressor windows, candidate visibility and source reuse; document-global query positions under contiguous CP; separate main/index QAT; floating masters and frozen published bytes", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/csa2.py",
        "api_and_state": "CSA2Config, AttentionState, CSA2Attention, Compressor, Indexer; explicit cache ownership and positions",
        "details": read_source("megatron/lite/primitive/modules/csa2.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use explicit compressed/sparse attention contract; full-document score materialization is a correctness provider, not fused sparse performance", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/deepseek_v41/test_attention.py, test_context_parallel.py and test_pp_stages.py; forward/VJP and owner/reindex boundaries", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
