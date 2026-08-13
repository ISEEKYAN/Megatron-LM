# Standalone Megatron FSDP Skill

Define, compose, and validate the standalone M-FSDP optimizer and its two
offload lifecycles.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.optimizer.mfsdp", kind="primitive", purpose="define and validate standalone M-FSDP",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def mfsdp(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.mfsdp, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("M-FSDP contract not satisfied", evidence=contract)

    principle = {
        "semantics": "match MCore M-FSDP parameter, gradient, and optimizer-shard ownership",
        "invariants": [
            "gather materializes reference weights before compute",
            "reduce-scatter lands in the owned sharded main gradient",
            "optimizer update matches the independent unsharded reference",
            "temporary communication storage has a bounded owner and lifetime",
        ],
        "reference": reference or "MCore M-FSDP lifecycle plus unsharded AdamW update",
    }
    implementation_contract = {
        "owned_files": ["primitive.optimizers.mfsdp", "primitive.optimizers.fp32_adamw"],
        "training_offload": [
            "CPU-resident local parameter shard and FP32 Adam master/moments",
            "bounded CPU-to-GPU parameter lease before each gather",
            "bounded GPU-to-CPU gradient staging during optimizer step",
        ],
        "rollout_offload": [
            "drain communication before transition",
            "release model, optimizer, and gradient GPU storage atomically",
            "restore fresh training gradients without retaining a CPU gradient copy",
        ],
        "boundaries": [
            "model code classifies ownership; optimizer primitive owns sharding and offload",
            "unsupported optimizer algorithms fail loudly instead of silently changing algorithm",
        ],
    }
    usage_contract = {
        "config": require_config_keys(config, ["optimizer", "offload_fraction"]),
        "choose_when": ["standalone M-FSDP is selected", "training or rollout GPU reclaim is required"],
        "valid": ["offload_fraction in [0, 1]", "AdamW for CPU update"],
        "avoid_when": ["the requested optimizer lacks an explicit CPU implementation"],
    }
    validation = primitive.validate(task, primitive=implementation.mfsdp, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("M-FSDP validation failed", evidence=validation)
    return done(
        principle=principle,
        implementation_contract=implementation_contract,
        usage_contract=usage_contract,
        validation=validation,
        risks=["state drift after DCP resume", "unbounded transfer staging", "stale rollout state"],
    )
```
