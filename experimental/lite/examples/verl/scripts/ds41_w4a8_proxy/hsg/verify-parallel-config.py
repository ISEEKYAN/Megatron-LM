"""Parse both resolved engine configs into the actual DS41 implementation config."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from hydra.utils import instantiate
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import _build_impl_cfg
from megatron.lite.runtime.contracts import ParallelConfig
from omegaconf import OmegaConf

source = Path(sys.argv[1])
raw = yaml.load(source.read_text(), Loader=yaml.CSafeLoader)
expected = {
    'pp': int(os.environ['DS41_PP']),
    'ep': int(os.environ.get('DS41_EP', '8')),
    'cp': int(os.environ.get('DS41_CP', '1')),
}
split = [
    int(cut)
    for cut in os.environ.get(
        'DS41_PIPELINE_SPLIT_LAYER', '2' if os.environ['DS41_LAYERS'] == '4' else ''
    ).split(',')
    if cut
]
split = None if not split else split[0] if len(split) == 1 else tuple(split)
rows = {}
actor_impl = None
for role in ('actor', 'ref'):
    engine_dict = raw['actor_rollout_ref'][role]['engine']
    # Parsing topology does not require expanding the large token map in OmegaConf.
    token_map = engine_dict['impl_cfg'].pop('token_map', None)
    engine = instantiate(OmegaConf.create(engine_dict))
    runtime = MegatronLiteConfig(
        parallel=ParallelConfig(
            tp=engine.tp,
            etp=engine.etp or 1,
            ep=engine.ep,
            pp=engine.pp,
            vpp=engine.vpp,
            cp=engine.cp,
        ),
        impl_cfg=(
            OmegaConf.to_container(engine.impl_cfg)
            if OmegaConf.is_config(engine.impl_cfg)
            else dict(engine.impl_cfg)
        ),
    )
    if token_map is not None:
        runtime.impl_cfg['token_map'] = token_map
    impl = _build_impl_cfg(protocol, runtime)
    if role == 'actor':
        actor_impl = impl
    actual = {key: getattr(impl.parallel, key) for key in expected}
    assert actual == expected, (role, actual, expected)
    assert impl.pipeline_split_layer == split, (role, impl.pipeline_split_layer, split)
    rows[role] = {**actual, 'pipeline_split_layer': impl.pipeline_split_layer}
assert rows['actor'] == rows['ref']
assert not torch.cuda.is_initialized()
receipt = {'passed': True, 'typed_impl_configs': rows, 'model_constructed': False}
if os.environ.get('DS41_META_WORKER') != '1':
    (source.parent / 'parallel-config.json').write_text(
        json.dumps(receipt, indent=2) + '\n'
    )
print('DS41_TYPED_PARALLEL_CONFIG', json.dumps(receipt), flush=True)

if os.environ.get('DS41_META_WORKER') != '1':
    subprocess.run(
        [
            sys.executable,
            '-m',
            'torch.distributed.run',
            '--standalone',
            '--nproc-per-node=' + os.environ['DS41_WORLD'],
            __file__,
            str(source),
        ],
        env={**os.environ, 'DS41_META_WORKER': '1'},
        check=True,
    )
else:
    import torch.distributed as dist
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import parameter_groups
    from megatron.lite.primitive.modules.engram_lookup import EngramTable
    from megatron.lite.primitive.modules.native_fp32_linear import (
        configure_residual_projections,
    )
    from megatron.lite.primitive.parallel.state import init_parallel

    dist.init_process_group('gloo')
    try:
        ps = init_parallel(replace(actor_impl.parallel, etp=1))
        model_cfg = protocol.build_model_config(
            raw['actor_rollout_ref']['model']['path']
        )
        stages = protocol._pipeline_stages(model_cfg, actor_impl, actor_impl.parallel)
        model = protocol._construct_model(
            model_cfg,
            replace(actor_impl, device='meta'),
            ps,
            stages[ps.pp_rank],
            stages,
        )
        if actor_impl.text_only:
            for binding in model.parameter_bindings():
                if binding.role in ('vision', 'aligner', 'image_delimiter'):
                    binding.tensor.requires_grad_(False)
        configure_residual_projections(
            model, actor_impl.dtype, actor_impl.optimizer == 'muon', EngramTable
        )
        from megatron.lite.model.deepseek_v41.lite.checkpoint import _SPEC

        header_spec = importlib.util.spec_from_file_location(
            'ds41_checkpoint_headers', Path(__file__).with_name('checkpoint-headers.py')
        )
        header_gate = importlib.util.module_from_spec(header_spec)
        header_spec.loader.exec_module(header_gate)
        model_path = raw['actor_rollout_ref']['model']['path']
        headers = header_gate.read_headers(model_path)
        header_receipt = header_gate.check(model, model_path, _SPEC, headers)
        header_receipt.update(
            passed=True,
            global_rank=dist.get_rank(),
            pp_rank=ps.pp_rank,
            stage_range=span,
            payload_read=False,
        )
        (source.parent / f'checkpoint-headers-rank-{dist.get_rank()}.json').write_text(
            json.dumps(header_receipt, indent=2) + '\n'
        )
        print('DS41_CHECKPOINT_HEADERS', json.dumps(header_receipt), flush=True)
        leaves = sum(parameter.requires_grad for parameter in model.parameters())
        owners = len(parameter_groups(model, lr=1e-6))
        assert leaves == owners, (leaves, owners)
        counts = os.environ.get('DS41_PARAMETER_COUNTS', '')
        if counts:
            assert leaves == int(counts.split(',')[ps.pp_rank]), (
                ps.pp_rank,
                leaves,
                counts,
            )
        assert not torch.cuda.is_initialized()
        result = {
            'passed': True,
            'global_rank': dist.get_rank(),
            'pp_rank': ps.pp_rank,
            'world': dist.get_world_size(),
            'ep': ps.ep_size,
            'cp': ps.cp_size,
            'stage_range': span,
            'trainable_leaves': leaves,
            'optimizer_owners': owners,
            'model_constructed': 'meta',
            'weights_loaded': False,
            'optimizer_state_constructed': False,
        }
        (source.parent / f'model-counts-rank-{dist.get_rank()}.json').write_text(
            json.dumps(result, indent=2) + '\n'
        )
        print('DS41_META_STAGE_COUNTS', json.dumps(result), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
