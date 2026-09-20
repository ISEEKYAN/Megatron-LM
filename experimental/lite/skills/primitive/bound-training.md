# Bound Training Primitives

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.bound_training", kind="primitive",
    purpose="route bound HF storage, row memory, paired streams, vision, block codecs and owned optimizer/DDP",
    imports=["basic.constitution"], calls=["primitive.validate"],
    inputs=["task", "files", "reference", "budget"],
    outputs=["routes", "validation", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
# Paths are relative to megatron/lite/primitive and tests/unit, respectively.
ROUTES = {
    "config_fields.py": "deepseek_v41/test_redo_parity.py::test_real_pp2_boundary_restarts_attention_state",
    "ckpt/row_stream.py": "deepseek_v41/test_bound_row_export.py::test_bound_save_streams_rows_and_exact_masters_under_budget",
    "ckpt/hf_weights.py": "deepseek_v41/test_redo_export_contract.py::test_hf_export_obeys_external_quantized_storage",
    "ckpt/binding_records.py": "deepseek_v41/test_redo_bindings.py::test_parameter_bindings_require_exact_owner_inventory",
    "modules/engram_lookup.py": "deepseek_v41/test_redo_engram_residency.py::test_forward_does_not_mutate_or_release_storage",
    "modules/owner_row_transport.py": "deepseek_v41/test_redo_engram_residency.py::test_owner_transport_preserves_compact_rows_and_backward",
    "modules/row_memory_build.py": "deepseek_v41/test_redo_parity.py::test_real_pp2_boundary_restarts_attention_state",
    "modules/paired_stream.py": "deepseek_v41/test_redo_parity.py::test_real_pp2_boundary_restarts_attention_state",
    "modules/image_data.py": "deepseek_v41/test_redo_parity.py::test_image_tokens_backpropagate_into_vision_and_aligner",
    "modules/vision.py": "deepseek_v41/test_redo_parity.py::test_image_tokens_backpropagate_into_vision_and_aligner",
    "modules/vision_training.py": "deepseek_v41/test_redo_parity.py::test_image_tokens_backpropagate_into_vision_and_aligner",
    "modules/native_fp32_linear.py": "deepseek_v41/test_redo_codecs.py::test_cross_layer_indexer_fp8_projection",
    "quantization/mxfp8.py": "deepseek_v41/test_redo_codecs.py::test_cross_layer_indexer_fp8_projection",
    "quantization/mxfp4.py": "deepseek_v41/test_redo_codecs.py::test_fp4_codec_rounding_and_surface",
    "quantization/nvfp4.py": "deepseek_v41/test_redo_codecs.py::test_fp4_codec_rounding_and_surface",
    "optimizers/headwise_muon.py": "deepseek_v41/test_redo_ep_finalize.py::test_ep_finalize_matches_single_global_batch",
    "optimizers/owned_groups.py": "deepseek_v41/test_redo_parity.py::test_optimizer_two_steps_and_nonfinite_transaction",
    "optimizers/sinkhorn.py": "deepseek_v41/test_redo_parity.py::test_optimizer_two_steps_and_nonfinite_transaction",
    "optimizers/staged_update.py": "deepseek_v41/test_redo_parity.py::test_remote_nonfinite_skips_replicated_optimizer",
    "parallel/owned_ddp.py": "deepseek_v41/test_redo_parity.py::test_packed_loss_head_gradient_with_ddp_unused_detection",
    "modules/attention/cp.py": "deepseek_v41/test_redo_cp_loss.py::test_cp_loss_matches_global_token_weighted_ce",
    "parallel/cp.py": "deepseek_v41/test_redo_cp_loss.py::test_cp_loss_matches_global_token_weighted_ce",
    "train_step.py": "deepseek_v41/test_redo_cp_loss.py::test_cp_loss_matches_global_token_weighted_ce",
    "modules/experts.py": "deepseek_v41/test_redo_moe_dual_bias.py::test_dispatch_option_reaches_model_and_preserves_local_moe",
}


def bound_training(task, files, reference, budget):
    files = {str(path).removeprefix("experimental/lite/megatron/lite/primitive/") for path in files}
    routes = sorted(files & ROUTES.keys())
    if not routes:
        return out_of_scope("no bound training primitive touched", evidence=sorted(files))
    if files - ROUTES.keys():
        return blocked("uncovered files require their own validation route", evidence=sorted(files - ROUTES.keys()))
    if reference is None:
        return blocked("require independent numerical or ownership reference")
    invariants = [
        "HF save/export reject unsupported resync symmetrically; archived bytes survive reload",
        "row owners retain storage; uint8 transport preserves FP8 scale bytes",
        "paired PP output and gradients equal the unpartitioned execution",
        "image tokens execute vision/aligner with finite nonzero gradients and updates",
        "default quantized execution uses real codecs, bounded error and exact replay checks",
        "optimizer commits all owners or none; dense/expert/row reductions use their actual groups",
    ]
    validation = []
    for route in routes[:budget.max_candidates]:
        result = primitive.validate(task, primitive=route, implementation=load_source_and_tests(route, reference, invariants), budget=budget)
        validation.append(result)
        if not result.done:
            return blocked("primitive validation failed", evidence=validation)
    if len(validation) != len(routes):
        return blocked("validation budget did not cover every touched route", evidence=validation)
    smoke = require_real_gpu_smoke(task, cases=["multimodal", "quantized", "ep2", "cp2"], skip_is_failure=True)
    if not smoke.done:
        return blocked("GPU composition smoke failed", evidence=smoke)
    return done(routes=routes, validation=validation, risks=["unit tests alone do not certify engine e2e"])
```

This is an orchestration specification, not a standalone executable validator.
`load_source_and_tests` and `require_real_gpu_smoke` are host-supplied, undefined
here. The mocked control-flow unit tests certify routing and failure propagation
only; their `done` does not certify numerical checks, GPU execution, or engine e2e.
An actual run needs independent test results and non-skipped GPU smoke evidence.

V4.1 release notes: archival HF save/export deliberately reject resync options;
CP loss deliberately fails without the prepared global normalization denominator.
Neither unsupported request is treated as a successful no-op.
