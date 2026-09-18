# Ds41 Fp8 Primitive Skill

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.quantization.ds41_fp8", kind="primitive", purpose="define and validate ds41 fp8",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def ds41_fp8(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("primitive contract not satisfied", evidence=contract)
    principle = {"invariants": "SWA row quantization differs from linear block32 quantization; preserve scale bytes, accumulation dtype and FP32 owner gradients", "reference": reference}
    implementation_contract = {
        "owned_file": "megatron/lite/primitive/quantization/ds41_fp8.py",
        "api_and_state": "FP8Values, quantize_swa, fake_quant_swa, dynamic_fp8_linear",
        "details": read_source("megatron/lite/primitive/quantization/ds41_fp8.py"),
    }
    usage_contract = {"selection_and_boundaries": "Use explicit activation/linear FP8 policy; do not merge with main-KV or index codecs", "config": config}
    validation = primitive.validate(task, primitive=implementation, implementation=implementation, budget=budget)
    if not validation.done:
        return blocked("primitive validation failed", evidence=validation)
    return done(
        principle=principle, implementation_contract=implementation_contract,
        usage_contract=usage_contract, validation=validation,
        risks={"required_checks": "tests/unit/deepseek_v41/test_quantization.py; compare FP8 GEMM to dequantized blockwise arithmetic", "limits": "CPU-only checks do not prove GPU or distributed correctness"},
    )
```
