# Standalone M-FSDP lifecycle translation ledger

Reference: `NVIDIA/Megatron-LM upstream/dev@43124b60`.

This ledger records the intentionally ported lifecycle subset for standalone
`optim_grads_params`. It is not a claim that Lite implements MCore HSDP,
DTensor, FP8 buffers, fine-grained hooks, or multiple distributed-optimizer
instances.

## Root backward lifecycle

- MCore `megatron_fsdp.py:926` `_root_pre_backward`
- Lite `buffer.py` `CommunicationPipelines.begin_backward` and
  `wrapper.py` `_BeginBackward.backward`
- Invariant: one GraphTask initializes one generation, one complete pending
  parameter set, and one queued root callback.
- Difference: Lite uses the PyTorch execution engine callback directly instead
  of MCore's composable FSDP state object.
- Proof: `test_mfsdp_begin_backward_is_once_only_until_post_backward_reset`,
  `test_mfsdp_multi_output_graph_runs_one_root_lifecycle`.

- MCore `megatron_fsdp.py:844` `_root_post_backward`
- Lite `buffer.py` `CommunicationPipelines.end_backward`
- Invariant: owners unresolved by unit hooks are processed in the same global
  reduction-group order and the pending parameter set must be empty.
- Proof: `test_mfsdp_root_post_backward_zeros_unused_params_and_exhausts_pending`,
  `test_mfsdp_unit_post_backward_defers_later_groups_until_order_is_stable`.

## Unit gradient processing

- MCore `megatron_fsdp.py:696` `_process_post_backward_gradients`
- Lite `buffer.py` `CommunicationPipelines.process_post_backward`
- Invariant: used gradients are staged in FP32, unused members are zero-filled,
  complete dense/expert groups become ready, and parameters are released only
  after their backward kernels.
- Difference: Lite's per-parameter post-accumulate hook only stages data. A
  module full-backward boundary or the root callback owns group processing.
- Proof: `test_mfsdp_grad_hook_only_stages_and_never_launches_collectives`,
  `test_mfsdp_root_post_backward_zeros_unused_params_and_exhausts_pending`.

- MCore `megatron_fsdp.py:792` `_register_post_backward_hook`
- Lite `wrapper.py` `_ReleaseBackward` plus the registered full-backward hook.
- Invariant: a unit can complete early only after at least one current-generation
  parameter notification; graph-root modules whose hooks fire early remain for
  root processing.
- Proof: multi-microbatch parity and two-rank Gloo parity in `test_mfsdp.py`.

## Deterministic reduce-scatter authority

- MCore `param_and_grad_buffer.py:4054`
  `get_ready_bucket_group_for_reduction`
- Lite `buffer.py` `GradReducePipeline.mark_group_ready`
- Invariant: `(owner_id, is_expert)` groups never mix process groups. A later
  group waits behind any unresolved earlier group.
- Difference: Lite makes the backward group order explicit and stable
  (reverse owner order, ascending bucket ID inside a group) to cover
  rank-local used/unused differences.
- Proof: `test_mfsdp_dense_and_expert_buckets_use_distinct_reduction_groups`,
  `test_mfsdp_unit_post_backward_defers_later_groups_until_order_is_stable`.

- MCore `param_and_grad_buffer.py:4087`
  `_bucket_group_gradient_reduce`
- Lite `buffer.py` `GradReducePipeline._drain_ready_groups` and
  `GradReducePipeline._reduce_bucket`
- Invariant: `_reduce_bucket` is the only production
  `reduce_scatter_tensor` call site. Parameter hooks cannot launch a
  collective.
- Proof: static call-site check plus the lifecycle tests above.

## Parameter gather and training state

- MCore `megatron_fsdp.py:888` `_pre_backward_param_unshard`
- Lite `wrapper.py` `_AcquireBackward` and
  `buffer.py` `AllGatherPipeline.acquire_backward_ids`
- Invariant: backward consumes materialized parameters and reverse-order
  prefetch remains bucket-scoped.
- Proof: saved-view, all-gather ordering, and recompute tests.

- MCore `megatron_fsdp.py:967` `_post_forward`
- Lite `buffer.py` `CommunicationPipelines.release_forward_owner`
- Invariant: ordinary forward reshards immediately; `PRE_BACKWARD` recompute
  keeps the lease until the corresponding backward release.
- Difference: standalone non-unit parameters use the same explicit
  gather/release policy as root-owned buckets. This avoids retaining shared
  allocator scratch across colocated export/wake boundaries.
- Proof: `test_activation_recompute_uses_pre_backward_lazy_release` and all
  tests in `test_mfsdp_recompute_forward.py`.

## Reset and terminal paths

- MCore `param_and_grad_buffer.py:3934,4413` reset paths
- Lite `CommunicationPipelines.finish_grad_sync`, `abort`, and
  `_quiesce_transient_state`
- Invariant: issued work is drained without launching from a wait path;
  pending groups, generation/readiness, full-param/main-grad leases, and
  training state are closed before export or device movement.
- Difference: because failed PyTorch GraphTasks do not reliably run queued
  callbacks, the next reachable top-level forward detects and aborts a stale
  generation.
- Proof: `test_mfsdp_backward_exception_is_torn_down_by_next_forward`,
  `test_mfsdp_release_scratch_drains_pending_lifecycle`, move/offload tests.
