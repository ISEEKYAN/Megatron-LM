"""Run the pinned VERL main GRPO loop through its existing MLite compat launcher."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from recipe_contract import hydra_mapping, rollout_settings

root = Path(__file__).resolve().parent
out = Path(os.environ['DS41_OUTPUT'])
model = Path(os.environ['DS41_MODEL'])
rollout = rollout_settings(json.loads((model / 'config.json').read_text()), os.environ)
if rollout['breakable']:
    os.environ['VLLM_USE_BREAKABLE_CUDAGRAPH'] = '1'
ep = int(os.environ.get('DS41_EP', '8'))
cp = int(os.environ.get('DS41_CP', '1'))
pp = int(os.environ.get('DS41_PP', '1'))
world = int(os.environ.get('DS41_WORLD', str(ep * cp)))
# Prompts per step; each of the world's data-parallel ranks needs >=1 sequence.
train_batch = int(os.environ.get('DS41_TRAIN_BATCH', '8'))
kv_cache_bytes = int(os.environ.get('DS41_KV_CACHE_BYTES', str(256 * 2**20)))
segmented_host = '1'
# Frozen Engram tables in host memory, shared by the colocated actor and ref.
engram_host = os.environ.get('DS41_ENGRAM_HOST', '0')
if engram_host not in ('0', '1'):
    raise ValueError('DS41_ENGRAM_HOST must be 0 or 1')
# Training MoE transport (DS4 recipe: ACTOR_MOE_DISPATCHER=deepep) and the
# rollout all2all backend (DS4 aligned mode: deepep_low_latency).
dispatcher = os.environ.get('ACTOR_MOE_DISPATCHER', 'alltoall')
if dispatcher not in ('alltoall', 'deepep'):
    raise ValueError('ACTOR_MOE_DISPATCHER must be alltoall or deepep')
use_deepep = 'true' if dispatcher == 'deepep' else 'false'
# DeepEP dispatch VJP reduced as all-to-all does (bitwise dX across dispatchers).
routed_backward = os.environ.get('DEEPEP_ROUTED_BACKWARD', '0')
if routed_backward not in ('0', '1'):
    raise ValueError('DEEPEP_ROUTED_BACKWARD must be 0 or 1')
routed_backward = 'true' if use_deepep == 'true' and routed_backward == '1' else 'false'
all2all = os.environ.get('ROLLOUT_ALL2ALL_BACKEND', 'allgather_reducescatter')
if all2all not in ('allgather_reducescatter', 'deepep_low_latency'):
    raise ValueError(
        'ROLLOUT_ALL2ALL_BACKEND must be allgather_reducescatter or deepep_low_latency'
    )
bi = os.environ.get('VLLM_BATCH_INVARIANT', '1')
if bi not in ('0', '1'):
    raise ValueError('VLLM_BATCH_INVARIANT must be 0 or 1')
layers = int(os.environ.get('DS41_LAYERS', '2'))
# Stage cuts (comma-separated for PP>2); unset uses the topology's default cut
# (the candidate publisher: 20 for 40L, 6 for the 9L subset). The 4L prefix has
# no publisher, so it defaults to the cut before its owner layer 2.
pp_cuts = os.environ.get('DS41_PIPELINE_SPLIT_LAYER', '2' if layers == 4 else '')
pp_cuts = [int(cut) for cut in pp_cuts.split(',') if cut]
prompt_length = int(
    os.environ.get(
        'DS41_PROMPT_LENGTH',
        {4: '1536', 6: '17408', 8: '17408', 9: '17408'}.get(layers, '256'),
    )
)
response_length = int(os.environ.get('DS41_RESPONSE_LENGTH', '128'))
context_length = prompt_length + response_length
if layers not in (2, 4, 6, 8, 9, 40) or min(prompt_length, response_length) <= 0:
    raise ValueError(
        'Expected a 2L/4L prefix, a 6L/8L/9L release subset or the full 40L model '
        'and positive prompt/response lengths'
    )
if min(pp, ep, cp, world) < 1 or world % (ep * pp) or world % (cp * pp):
    raise ValueError(
        'DS41_PP must be positive and DS41_WORLD divisible by EP*PP and CP*PP'
    )
if pp_cuts and len(pp_cuts) != pp - 1:
    raise ValueError('DS41_PIPELINE_SPLIT_LAYER needs DS41_PP - 1 cuts')
split_layer = (
    {
        'pipeline_split_layer': (
            ','.join(map(str, pp_cuts)).join('[]')
            if len(pp_cuts) > 1
            else str(pp_cuts[0])
        )
    }
    if pp > 1 and pp_cuts
    else {}
)
if train_batch < 1 or (train_batch * 4) % (world // cp):
    raise ValueError(
        'DS41_TRAIN_BATCH x n=4 must divide evenly over data-parallel ranks'
    )
if not os.environ.get('DS41_DATA'):
    raise ValueError('DS41_DATA must point to the DAPO training parquet')
os.environ['W4_GRPO_DATA'] = os.environ['DS41_DATA']
os.environ['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID', 'local')
token_map = json.loads((model / 'token-map.json').read_text())
runtime = out / 'hydra-config'
(runtime / 'critic').mkdir(parents=True, exist_ok=True)
# Keep every upstream field; pin the three extension config choices explicitly.
verl_spec = importlib.util.find_spec('verl')
if verl_spec is None or verl_spec.origin is None:
    raise RuntimeError('VERL must be installed in the recipe image')
upstream = Path(verl_spec.origin).resolve().parent / 'trainer/config'
primary = (upstream / 'ppo_trainer.yaml').read_text()
for group, name in [
    ('actor@actor_rollout_ref.actor', 'mlite_actor'),
    ('ref@actor_rollout_ref.ref', 'mlite_ref'),
    ('critic@critic', 'mlite_critic'),
]:
    primary = primary.replace(
        f'{group}: ${{model_engine}}_' + group.split('@')[0], f'{group}: {name}'
    )
from omegaconf import OmegaConf

primary_config = OmegaConf.create(primary)
OmegaConf.update(
    primary_config,
    'actor_rollout_ref.actor.engine.impl_cfg.token_map',
    token_map,
    force_add=True,
)
OmegaConf.update(
    primary_config,
    'actor_rollout_ref.ref.engine.impl_cfg.token_map',
    token_map,
    force_add=True,
)
# Explicit plain-text prompt rendering; the true tokenizer files stay exact.
OmegaConf.update(
    primary_config,
    'actor_rollout_ref.model.custom_chat_template',
    "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}",
    force_add=True,
)
OmegaConf.save(primary_config, runtime / 'w4-grpo.yaml')
(runtime / 'critic/mlite_critic.yaml').write_text(
    '_target_: verl.workers.config.CriticConfig\nenable: false\nstrategy: mlite\n'
)
args = [
    f'hydra.searchpath=[file://{upstream},pkg://verl_mlite.config]',
    '~actor_rollout_ref.actor.engine.grad_offload',
    '~actor_rollout_ref.ref.engine.grad_offload',
    'algorithm.adv_estimator=grpo',
    'algorithm.norm_adv_by_std_in_grpo=true',
    'algorithm.use_kl_in_reward=false',
    'algorithm.rollout_correction.bypass_mode=false',
    'algorithm.rollout_correction.rollout_is=null',
    'algorithm.rollout_correction.rollout_rs=null',
    f'data.train_files={os.environ["W4_GRPO_DATA"]}',
    f'data.val_files={os.environ["W4_GRPO_DATA"]}',
    f'data.train_batch_size={train_batch}',
    f'data.max_prompt_length={prompt_length}',
    f'data.max_response_length={response_length}',
    'data.truncation=left',
    'data.filter_overlong_prompts=false',
    'data.seed=42',
    'data.return_raw_chat=true',
    f'actor_rollout_ref.model.path={model}',
    'actor_rollout_ref.model.trust_remote_code=true',
    'actor_rollout_ref.model.use_remove_padding=true',
    'actor_rollout_ref.model.use_fused_kernels=false',
    f'actor_rollout_ref.actor.ppo_mini_batch_size={train_batch}',
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1',
    'actor_rollout_ref.actor.use_dynamic_bsz=false',
    'actor_rollout_ref.actor.ppo_epochs=1',
    'actor_rollout_ref.actor.shuffle=false',
    'actor_rollout_ref.actor.entropy_coeff=0',
    'actor_rollout_ref.actor.calculate_entropy=false',
    'actor_rollout_ref.actor.use_kl_loss=true',
    'actor_rollout_ref.actor.kl_loss_coef=0.001',
    'actor_rollout_ref.actor.kl_loss_type=low_var_kl',
    'actor_rollout_ref.actor.clip_ratio=0.2',
    'actor_rollout_ref.actor.clip_ratio_low=0.2',
    'actor_rollout_ref.actor.clip_ratio_high=0.2',
    'actor_rollout_ref.actor.loss_agg_mode=token-mean',
    'actor_rollout_ref.actor.optim.lr=1e-6',
    'actor_rollout_ref.actor.optim.lr_warmup_steps=0',
    'actor_rollout_ref.actor.engine.model_name=deepseek_v41',
    'actor_rollout_ref.actor.engine.impl=lite',
    'actor_rollout_ref.actor.engine.load_hf_weights=true',
    'actor_rollout_ref.actor.engine.attention_backend_override=null',
    'actor_rollout_ref.actor.engine.param_offload=true',
    'actor_rollout_ref.actor.engine.optimizer_offload=false',
    '+actor_rollout_ref.actor.engine.full_determinism=true',
    '+actor_rollout_ref.actor.engine.seed=42',
    f'actor_rollout_ref.actor.engine.ep={ep}',
    f'actor_rollout_ref.ref.engine.ep={ep}',
    f'actor_rollout_ref.actor.engine.pp={pp}',
    f'actor_rollout_ref.ref.engine.pp={pp}',
    'actor_rollout_ref.actor.engine.resync_format=mxfp4',
    # Two frozen tables (6L/8L/9L subsets, 40L): resend them every resync so
    # the rollout's level-2 sleep keeps no per-rank host copy (47.2 GiB/rank).
    '++actor_rollout_ref.actor.engine.resync_config={expert_dtype:fp4,freeze_engram:true'
    + ('}' if layers in (2, 4) else ',resend_frozen_engram:true}'),
    *[
        f'++actor_rollout_ref.actor.engine.impl_cfg.{k}={v}'
        for k, v in {
            'quantized': 'true',
            'w4a8_experts': 'true',
            'deployment_math': 'true',
            'trainable_engram': 'false',
            **({'host_engram': 'true'} if engram_host == '1' else {}),
            'use_deepep': 'false',
            'use_deepep': use_deepep,
            'deepep_routed_backward': routed_backward,
            'dtype': 'bfloat16',
            **split_layer,
            'optimizer': 'muon',
            'optimizer_config': '{lr:1e-6,ns_steps:2,coefficient_type:quintic,segmented_host:'
            + ('true' if segmented_host == '1' else 'false')
            + '}',
        }.items()
    ],
    'actor_rollout_ref.ref.engine.load_hf_weights=true',
    'actor_rollout_ref.ref.engine.param_offload=true',
    'actor_rollout_ref.ref.engine.attention_backend_override=null',
    *[
        f'++actor_rollout_ref.ref.engine.impl_cfg.{k}={v}'
        for k, v in {
            'quantized': 'true',
            'w4a8_experts': 'true',
            'deployment_math': 'true',
            'trainable_engram': 'false',
            **({'host_engram': 'true'} if engram_host == '1' else {}),
            'use_deepep': 'false',
            'use_deepep': use_deepep,
            'deepep_routed_backward': routed_backward,
            'dtype': 'bfloat16',
            **split_layer,
        }.items()
    ],
    'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1',
    '+actor_rollout_ref.ref.engine.full_determinism=true',
    '+actor_rollout_ref.ref.engine.seed=42',
    'actor_rollout_ref.rollout.name=vllm',
    'actor_rollout_ref.rollout.full_determinism=true',
    'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
    f'actor_rollout_ref.rollout.data_parallel_size={world}',
    f'actor_rollout_ref.rollout.expert_parallel_size={world}',
    'actor_rollout_ref.rollout.agent.num_workers=4',
    'actor_rollout_ref.rollout.n=4',
    f'actor_rollout_ref.rollout.agent.agent_loop_config_path={root}/seeded-agent.yaml',
    'actor_rollout_ref.rollout.agent.default_agent_loop=ds41_seeded_single_turn',
    'actor_rollout_ref.rollout.temperature=1.0',
    'actor_rollout_ref.rollout.top_p=1.0',
    'actor_rollout_ref.rollout.top_k=-1',
    'actor_rollout_ref.rollout.ignore_eos=true',
    'actor_rollout_ref.rollout.calculate_log_probs=true',
    'actor_rollout_ref.rollout.logprobs_mode=raw_logprobs',
    'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1',
    f'actor_rollout_ref.rollout.max_model_len={context_length}',
    'actor_rollout_ref.rollout.max_num_seqs=4',
    f'actor_rollout_ref.rollout.max_num_batched_tokens={context_length}',
    'actor_rollout_ref.rollout.gpu_memory_utilization=0.2',
    f'actor_rollout_ref.rollout.enforce_eager={str(rollout["enforce_eager"]).lower()}',
    'actor_rollout_ref.rollout.enable_prefix_caching=false',
    'actor_rollout_ref.rollout.load_format=dummy',
    'actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=verl_mlite.rollout.deepseek_v41.DeepseekV41WorkerExtension',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASHMLA_SPARSE_DSV41',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=nvfp4_ds_mla',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config={indexer_kv_dtype:mxfp4,indexer_sparse_logits:false}',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides='
    + hydra_mapping(rollout['hf_overrides']),
    f'+actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend={all2all}',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=deep_gemm',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config='
    + hydra_mapping(rollout['compilation_config']),
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.engram_config={cpu_offload:true,embedding_across_dp:true,dp_shared_memory:false,host_table_layout:per_rank}',
    # Use the exact multi-layer replay receptive field from the vLLM overlay.
    # Leave swa_bounded_replay unset; the production resolver checks eligibility.
    f'+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes={kv_cache_bytes}',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:0}',
    'critic.enable=false',
    'reward.num_workers=4',
    'reward.reward_manager.name=naive',
    f'reward.custom_reward_function.path={root}/reward.py',
    'reward.custom_reward_function.name=compute_score',
    'trainer.logger=[console,file,wandb]',
    'trainer.project_name=ds41-w4a8-grpo-strict0',
    f'trainer.experiment_name=ds41-w4a8-bi{bi}-{os.environ["SLURM_JOB_ID"]}',
    f'trainer.n_gpus_per_node={os.environ.get("DS41_GPUS_PER_NODE", "4")}',
    f'trainer.nnodes={world//int(os.environ.get("DS41_GPUS_PER_NODE", "4"))}',
    f'trainer.total_training_steps={os.environ.get("DS41_STEPS", "2")}',
    f'trainer.save_freq={os.environ.get("DS41_SAVE_FREQ", "2")}',
    'trainer.test_freq=-1',
    'trainer.val_before_train=false',
    'trainer.use_v1=false',
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]',
    f'trainer.default_local_dir={out}/checkpoints',
    f'hydra.run.dir={out}/hydra',
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_MLITE_SKIP_RUNTIME_PATCHES="0"',
    f'+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT="{bi}"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_MLITE_HF_CONFIG_MODEL_TYPE=deepseek_v41',
    '+ray_kwargs.ray_init.runtime_env.env_vars.MEGATRON_LITE_MOE_PERMUTE_FUSION="0"',
    f'+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_FINISH_TIMEOUT="{os.environ.get("WANDB_FINISH_TIMEOUT", "60")}"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_FINISH_TIMEOUT_RAISES="false"',
    '++ray_kwargs.ray_init.address=' + os.environ.get('DS41_RAY_ADDRESS', 'auto'),
    f'ray_kwargs.ray_init.runtime_env.py_executable={sys.executable}',
    '+ray_kwargs.ray_init.runtime_env.env_vars.OMEGACONF_MAX_YAML_EXPANDED_NODES="600000"',
    f'+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="{root}/audit:{os.environ.get("PYTHONPATH", "")}"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.W4_GRPO_AUDIT="1"',
    f'+ray_kwargs.ray_init.runtime_env.env_vars.W4_GRPO_OUT={out}',
]
for key in (
    'VLLM_USE_BREAKABLE_CUDAGRAPH',
    'DS41_PRECISION',
    'DS41_CUDAGRAPH',
    'DS41_CUDAGRAPH_MAX_CAPTURE',
    'DS41_INDEXER_DIAGNOSTIC',
    'DS41_INDEXER_DIAGNOSTIC_LIMIT',
    'WANDB_ENTITY',
    'WANDB_DIR',
    'WANDB_CACHE_DIR',
    'WANDB_DATA_DIR',
    'WANDB_CONFIG_DIR',
    'TMPDIR',
    'TMP',
    'TEMP',
    'DG_JIT_CACHE_DIR',
    'TRITON_CACHE_DIR',
    'TORCHINDUCTOR_CACHE_DIR',
    'TORCH_EXTENSIONS_DIR',
    'CUDA_CACHE_PATH',
    'XDG_CACHE_HOME',
    'FLASHINFER_WORKSPACE_BASE',
    'CUTE_DSL_CACHE_DIR',
    'TILELANG_CACHE_DIR',
    'TILELANG_TMP_DIR',
    'HUMMING_CACHE_DIR',
    'HUMMING_TMP_DIR',
    'HUMMING_DISABLE_PARALLEL_BUILD',
):
    if key in os.environ:
        args.append(
            f'++ray_kwargs.ray_init.runtime_env.env_vars.{key}={json.dumps(os.environ[key])}'
        )
# Opt-in knobs append arguments only when set, so default runs keep the
# exact argument list of the validated 2L/4L recipe.
if cp != 1:
    args += [
        f'actor_rollout_ref.actor.engine.cp={cp}',
        f'actor_rollout_ref.ref.engine.cp={cp}',
    ]
if os.environ.get('DS41_RECOMPUTE'):
    # Long sequences: full activation recompute of CSA-state-consuming layers.
    args += [
        f'++actor_rollout_ref.{role}.engine.impl_cfg.recompute='
        + os.environ['DS41_RECOMPUTE']
        for role in ('actor', 'ref')
    ]
if os.environ.get('DS41_RESUME'):
    args += [
        'trainer.resume_mode=resume_path',
        'trainer.resume_from_path=' + os.environ['DS41_RESUME'],
    ]
else:
    args += ['trainer.resume_mode=disable']
command = [
    sys.executable,
    '-m',
    'verl_mlite.launch',
    'verl.trainer.main_ppo',
    f'--config-path={runtime}',
    '--config-name=w4-grpo',
    *args,
]
(out / 'command.json').write_text(json.dumps(command, indent=2))
(out / 'inputs.json').write_text(
    json.dumps(
        {
            'data_sha256': hashlib.file_digest(
                Path(os.environ['DS41_DATA']).open('rb'), 'sha256'
            ).hexdigest(),
            'model_config_sha256': hashlib.file_digest(
                (model / 'config.json').open('rb'), 'sha256'
            ).hexdigest(),
            'token_map_sha256': hashlib.file_digest(
                (model / 'token-map.json').open('rb'), 'sha256'
            ).hexdigest(),
            'batch_invariant': int(bi),
            'seed': 42,
            'layers': layers,
            'rollout_settings': rollout,
            'prompt_length': prompt_length,
            'response_length': response_length,
            'generator_imported_torch': 'torch' in sys.modules,
        },
        indent=2,
    )
    + '\n'
)
preflight = '--config-only' in sys.argv
execution_env = dict(
    os.environ,
    VERL_MLITE_SKIP_RUNTIME_PATCHES='0',
    W4_GRPO_AUDIT='0' if preflight else '1',
)
if preflight:
    (out / 'resolved.yaml').write_text(
        subprocess.check_output(
            [*command, '--cfg', 'job', '--resolve'], text=True, env=execution_env
        )
    )
    print('GRPO_HYDRA_CONFIG_OK ' + str(out / 'resolved.yaml'), flush=True)
else:
    raise SystemExit(subprocess.call(command, env=execution_env))
