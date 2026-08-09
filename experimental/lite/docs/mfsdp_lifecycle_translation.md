# Standalone M-FSDP lifecycle translation ledger

Reference: `NVIDIA/Megatron-LM upstream/dev@43124b60`.

This ledger is the release contract for standalone `optim_grads_params`.
Every in-scope row must end as `exact`, `equivalent-with-proof`, or
`unsupported-fail-fast`.  `divergent` and `missing` are implementation
blockers, not accepted release notes.

Out of scope for this delivery: FP8, HSDP, and CUDA Graph.  CPU optimizer
offload performance is tracked separately; offload checkpoint correctness
remains in scope.

## Frozen references and acceptance baseline

- Semantic reference: `NVIDIA/Megatron-LM upstream/dev@43124b60`.
- Product/performance reference: PR #89 head `15145be3a`.
- Public-state reference: Lite FSDP2 at this branch.
- Current repaired-overlap result (`ac4bd8140`, Qwen3.5 8-layer/8-expert,
  DP8, four microbatches, sequence length 1024, offload=0):
  - 20/20 finite updates; loss `12.4322357 -> 12.4297810`;
    grad norm `0.8542209 -> 0.8539507`.
  - `1357.37 ms/step`, `24140.86 tok/s`, `19.205 GB` peak allocated.
- Same-shape FSDP2 archive:
  - `1219.07 ms/step`, `26879.49 tok/s`, `11.578 GB` peak allocated.
- Same-shape dist-opt archive:
  - `1032.31 ms/step`, `31742.44 tok/s`, `16.936 GB` peak allocated.
- PR #89's exact archived command and numbers must be recovered before final
  acceptance.  A differently shaped run is not a substitute.

Hard release gates: M-FSDP throughput must be no lower than FSDP2, peak
allocated memory no higher than FSDP2, and neither metric may regress the
same-command PR #89 archive.

## Full contract status

- Parameter flatten/pad/local shard: `equivalent-with-proof`.
- Dense/expert process-group selection: `equivalent-with-proof`.
- Custom gather-group rank-order validation: `equivalent-with-proof`; startup
  rejects any AG group whose global-rank order differs from its data group.
- Initial full-to-persistent-shard transition: `equivalent-with-proof`.
- Forward owner AG and pure-DP prefetch: `equivalent-with-proof`.
- AG ProcessGroup Work lifetime: `equivalent-with-proof` after `71e33aef6`.
- AG overlap under intersecting TP/EP/ETP/CP groups:
  `equivalent-with-proof`; multidimensional configurations retain MCore's
  dedicated-stream overlap and no longer silently force the serial path.
- Double-buffer pool layout: `equivalent-with-proof`; dtype/bucket-offset max
  slots are shared across unit layouts and pool exhaustion uses MCore's dynamic
  backup allocation.
- Non-double-buffer AG/RS live-set bound: `equivalent-with-proof`; the default
  is bounded by two maximum owner groups, while an explicit override remains
  available.
- Root/non-unit dynamic fallback allocation: `equivalent-with-proof`.
- Unit post-forward reshard: `equivalent-with-proof`.
- Cross-unit tied/shared parameters: `equivalent-with-proof`; all bindings are
  discovered with `remove_duplicate=False`, promoted to the root owner, and
  represented by one reduction spec.
- Saved-tensor parameter-view restoration: `equivalent-with-proof`.
- Recompute `PRE_BACKWARD` release: `equivalent-with-proof`.
- Ordinary and TE GroupedLinear authoritative FP32 gradient destination:
  `equivalent-with-proof`; both write the parameter-group FP32 bucket directly.
- TE delayed-wgrad callback: `equivalent-with-proof`; parameters marked
  `skip_backward_post_hook` receive MCore's post-wgrad completion callback.
- Globally stable RS ordering: `equivalent-with-proof`.
- RS Work completion: `equivalent-with-proof`; normal paths install stream
  dependencies and consume ProcessGroup Work without host event synchronization.
- Multi-microbatch FP32 shard accumulation: `equivalent-with-proof`.
- Optimizer step to next AG: `equivalent-with-proof`.
- Eval-to-train overlap lifecycle: `equivalent-with-proof`, GPU job 15407429.
- Public `named_parameters` and `state_dict`: `equivalent-with-proof`; both
  expose persistent optimizer shards and never call `materialize_all`.
- Model DCP persistent distributed state: `equivalent-with-proof` for the same
  topology. Each parameter is a DP-group `DTensor` over its authoritative
  flat shard; group identity and VPP chunk index are part of the key.
  Cross-topology M-FSDP restore is `unsupported-fail-fast` because changing
  DP-group membership changes the key/global padded shape.
- Model-only restore and CPU-master refresh: `equivalent-with-proof`.
- Bounded HF export: `equivalent-with-proof`.
- Forward/backward exception teardown: `equivalent-with-proof`; normal finish,
  abort, export, persistent-state access, and device move share closed
  lifecycle bookkeeping, while exception teardown host-synchronizes issued
  work before releasing leases.
- Device move: `equivalent-with-proof` for current offload contract.
- PP/VPP chunk state and checkpoint-key uniqueness: `equivalent-with-proof`;
  checkpoint keys include PP stage, VPP chunk, canonical parameter name, and
  data-group identity.
- Full DP/TP/EP/ETP/PP/VPP/CP topology contract: implementation is
  `equivalent-with-proof`; the combined GPU evidence gate remains required.

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
