# M-FSDP Distributed Checkpoint Skill

Define and validate distributed checkpoint continuity for standalone M-FSDP,
including CPU-offloaded AdamW state.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.checkpoint.mfsdp", kind="primitive", purpose="checkpoint standalone M-FSDP state",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate", "primitive.optimizer.mfsdp"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def mfsdp(task, implementation, config, reference, budget):
    optimizer = primitive.optimizer.mfsdp(task, implementation, config, reference, budget)
    if not optimizer.done:
        return blocked("M-FSDP optimizer contract not satisfied", evidence=optimizer)
    contract = primitive.contract(implementation.checkpoint, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("M-FSDP checkpoint contract not satisfied", evidence=contract)

    principle = {
        "semantics": "save-load-continue matches uninterrupted M-FSDP training",
        "state": [
            "model parameter shards",
            "GPU and CPU optimizer parameter-group metadata",
            "FP32 master parameters, exp_avg, exp_avg_sq, and step",
        ],
        "reference": reference or "uninterrupted next optimizer step",
    }
    implementation_contract = {
        "owned_files": ["primitive.ckpt.dcp", "primitive.optimizers.mfsdp"],
        "placement": [
            "CPU-resident shards stage through the process-group compute device",
            "restored pinned tensors preserve their allocated storage and parameter identities",
        ],
        "failure_modes": [
            "missing GPU or CPU optimizer metadata fails loudly",
            "bucket identity or parameter-group count mismatch fails loudly",
        ],
    }
    usage_contract = {
        "choose_when": ["optimizer backend is mfsdp", "runtime use_dcp is true"],
        "valid": ["model-only", "optimizer-only", "combined", "same or changed DP size"],
        "avoid_when": ["loading CPU-offloaded state into a target without the matching offload contract"],
    }
    validation = primitive.validate(task, primitive=implementation.checkpoint, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("M-FSDP checkpoint validation failed", evidence=validation)
    return done(
        principle=principle,
        implementation_contract=implementation_contract,
        usage_contract=usage_contract,
        validation=validation,
        risks=["silent optimizer-state omission", "DP-reshard metadata drift", "checkpoint-time staging peak"],
    )
```
