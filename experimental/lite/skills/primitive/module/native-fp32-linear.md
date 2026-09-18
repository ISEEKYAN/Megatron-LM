# Native Fp32 Linear Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.native_fp32_linear", kind="primitive", purpose="define and validate native fp32 linear",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def native_fp32_linear(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "Forward uses activation dtype; wgrad GEMM uses FP32 operands directly into FP32 master; preserve per-call quantization mode and reject non-FP32 master", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/modules/native_fp32_linear.py",
        "api_and_state": "native_fp32_linear, Linear, FP4Linear; FP32 weight owner and explicit quantization switches",
        "details": read_source("megatron/lite/primitive/modules/native_fp32_linear.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use native correctness provider when FP32 owner gradients are required; not a TE fused-performance claim", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/deepseek_v41/test_headwise_muon.py and test_quantization.py; dtype/update and quantized forward reference", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
