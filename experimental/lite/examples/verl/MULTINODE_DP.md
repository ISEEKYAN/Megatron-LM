# VERL single frontend with remote headless DP ranks

This adapter targets the VERL engine-worker API pinned in REQUIRED_VERL.txt
(including the #7688 constructor path). Load `verl_mlite.compat` using the
existing MLite launcher before constructing rollout servers. It does not
launch a cluster or allocate GPUs.

For two nodes with four GPUs each, retain the normal VERL GRPO configuration
and apply these overrides through the existing GRPO launcher:

```bash
NNODES=2 NGPUS_PER_NODE=4 \
  bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh \
  trainer.nnodes=2 trainer.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=8 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_expert_parallel=true \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend=allgather_reducescatter
```

Supply MODEL_PATH, TRAIN_FILES and VAL_FILES as described in README.md.
Use the same Ray cluster setup as the ordinary launcher. The example selects
an existing Qwen consumer; DS4.1 W4A8 additionally requires its model,
expert/deployment, resync and packed-head PRs and a supported vLLM fork.
This adapter itself does not guarantee numerical parity or performance.

VERL runs one API frontend on node0 and headless engines on other nodes.
For that default topology only, omit node0's `data_parallel_start_rank=0`
from a copied argument object so vLLM infers the head rank from node_rank.
Keep local/global DP sizes, the master port, expert-parallel arguments and
remote headless start ranks unchanged. Explicit hybrid/external LB modes,
single-node jobs and a nonzero node rank preserve their original arguments.
The patch is idempotent and does nothing when vLLM is not importable.

Remote MLite workers can load the VERL engine API without importing optional
GPU/rollout dependencies. Existing file-based engine loading remains the
fallback when the package import raises ImportError or ModuleNotFoundError.
Opaque HF config registration and scoped reload-buffer handling retain their
existing behavior.

Run the CPU compatibility contract with:

```bash
PYTHONPATH=experimental/lite python -m pytest \
  experimental/lite/tests/unit/examples/test_verl_compat_multinode_dp.py \
  experimental/lite/tests/unit/examples/test_verl_mlite_engine_config.py
```

Full two-node serving requires the actual pinned VERL/vLLM environment and
is separate from these argument/API unit tests.
