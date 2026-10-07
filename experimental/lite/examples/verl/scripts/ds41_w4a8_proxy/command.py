"""Run the pinned VERL main GRPO loop through its existing MLite compat launcher."""

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
out = Path(os.environ['DS41_OUTPUT'])
model = Path(os.environ['DS41_MODEL'])
ep = int(os.environ.get('DS41_EP', '8'))
pp = int(os.environ.get('DS41_PP', '1'))
if (pp, ep) not in ((1, 8), (2, 4)):
    raise ValueError('Proxy layouts are PP1/EP8 or PP2/EP4, world8')
world = 8
if not os.environ.get('DS41_DATA'):
    raise ValueError('DS41_DATA must point to the DAPO training parquet')
os.environ['W4_GRPO_DATA'] = os.environ['DS41_DATA']
os.environ['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID', 'local')
token_map = json.loads((model / 'token-map.json').read_text())
runtime = out / 'hydra-config'
(runtime / 'critic').mkdir(parents=True, exist_ok=True)
# Keep every upstream field; pin the three extension config choices explicitly.
import verl

upstream = Path(verl.__file__).resolve().parent / 'trainer/config'
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
    'data.train_batch_size=8',
    'data.max_prompt_length=256',
    'data.max_response_length=128',
    'data.truncation=left',
    'data.filter_overlong_prompts=false',
    'data.seed=42',
    'data.return_raw_chat=true',
    f'actor_rollout_ref.model.path={model}',
    'actor_rollout_ref.model.trust_remote_code=true',
    'actor_rollout_ref.model.use_remove_padding=true',
    'actor_rollout_ref.model.use_fused_kernels=false',
    'actor_rollout_ref.actor.ppo_mini_batch_size=8',
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
    '++actor_rollout_ref.actor.engine.impl_cfg.pipeline_split_layer=1',
    '++actor_rollout_ref.ref.engine.impl_cfg.pipeline_split_layer=1',
    'actor_rollout_ref.actor.engine.resync_format=mxfp4',
    '++actor_rollout_ref.actor.engine.resync_config={expert_dtype:fp4,freeze_engram:true}',
    *[
        f'++actor_rollout_ref.actor.engine.impl_cfg.{k}={v}'
        for k, v in {
            'quantized': 'true',
            'w4a8_experts': 'true',
            'deployment_math': 'true',
            'trainable_engram': 'false',
            'use_deepep': 'false',
            'dtype': 'bfloat16',
            'optimizer': 'muon',
            'optimizer_config': '{lr:1e-6,ns_steps:2,coefficient_type:quintic}',
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
            'use_deepep': 'false',
            'dtype': 'bfloat16',
        }.items()
    ],
    'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1',
    '+actor_rollout_ref.ref.engine.full_determinism=true',
    '+actor_rollout_ref.ref.engine.seed=42',
    'actor_rollout_ref.rollout.name=vllm',
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
    'actor_rollout_ref.rollout.max_model_len=384',
    'actor_rollout_ref.rollout.max_num_seqs=4',
    'actor_rollout_ref.rollout.max_num_batched_tokens=384',
    'actor_rollout_ref.rollout.gpu_memory_utilization=0.2',
    'actor_rollout_ref.rollout.enforce_eager=true',
    'actor_rollout_ref.rollout.enable_prefix_caching=false',
    'actor_rollout_ref.rollout.load_format=dummy',
    'actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=verl_mlite.rollout.deepseek_v41.DeepseekV41WorkerExtension',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASHMLA_SPARSE_DSV41',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=nvfp4_ds_mla',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config={indexer_kv_dtype:mxfp4}',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides={head_dtype:float32}',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.all2all_backend=allgather_reducescatter',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=deep_gemm',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.engram_config={cpu_offload:false,embedding_across_dp:true,dp_shared_memory:false}',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=268435456',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:0}',
    'critic.enable=false',
    'reward.num_workers=4',
    'reward.reward_manager.name=naive',
    f'reward.custom_reward_function.path={root}/reward.py',
    'reward.custom_reward_function.name=compute_score',
    'trainer.logger=[console,file,wandb]',
    'trainer.project_name=ds41-w4a8-grpo-strict0',
    f'trainer.experiment_name=w4-real-prefix-grpo-{os.environ["SLURM_JOB_ID"]}',
    f'trainer.n_gpus_per_node={os.environ.get("DS41_GPUS_PER_NODE", "4")}',
    f'trainer.nnodes={world//int(os.environ.get("DS41_GPUS_PER_NODE", "4"))}',
    f'trainer.total_training_steps={os.environ.get("DS41_STEPS", "20")}',
    f'trainer.save_freq={os.environ.get("DS41_SAVE_FREQ", "10")}',
    'trainer.test_freq=-1',
    'trainer.val_before_train=false',
    'trainer.use_v1=false',
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]',
    f'trainer.default_local_dir={out}/checkpoints',
    f'hydra.run.dir={out}/hydra',
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_MLITE_SKIP_RUNTIME_PATCHES="0"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT="1"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DEEP_GEMM_MEGA_MHC_USE_VENDORED="1"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DEEP_GEMM_PAGED_MQA_USE_VENDORED="1"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_MLITE_HF_CONFIG_MODEL_TYPE=deepseek_v41',
    '+ray_kwargs.ray_init.runtime_env.env_vars.MEGATRON_LITE_MOE_PERMUTE_FUSION="0"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY=megatron-core-moe-dev',
    '++ray_kwargs.ray_init.address=' + os.environ.get('DS41_RAY_ADDRESS', 'auto'),
    f'ray_kwargs.ray_init.runtime_env.py_executable={sys.executable}',
    '+ray_kwargs.ray_init.runtime_env.env_vars.OMEGACONF_MAX_YAML_EXPANDED_NODES="600000"',
    f'+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="{root}/audit:{os.environ.get("PYTHONPATH", "")}"',
    '+ray_kwargs.ray_init.runtime_env.env_vars.W4_GRPO_AUDIT="1"',
    f'+ray_kwargs.ray_init.runtime_env.env_vars.W4_GRPO_OUT={out}',
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
preflight = '--config-only' in sys.argv
if preflight:
    (out / 'resolved.yaml').write_text(
        subprocess.check_output([*command, '--cfg', 'job', '--resolve'], text=True)
    )
    print('GRPO_HYDRA_CONFIG_OK ' + str(out / 'resolved.yaml'), flush=True)
else:
    raise SystemExit(subprocess.call(command))
