"""Parse both resolved engine configs into the actual DS41 implementation config."""

import json
import os
import sys
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
split = int(
    os.environ.get(
        'DS41_PIPELINE_SPLIT_LAYER', '2' if os.environ['DS41_LAYERS'] == '4' else '20'
    )
)
rows = {}
for role in ('actor', 'ref'):
    engine_dict = raw['actor_rollout_ref'][role]['engine']
    # Parsing topology does not require expanding the large token map in OmegaConf.
    engine_dict['impl_cfg'].pop('token_map', None)
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
    impl = _build_impl_cfg(protocol, runtime)
    actual = {key: getattr(impl.parallel, key) for key in expected}
    assert actual == expected, (role, actual, expected)
    assert impl.pipeline_split_layer == split, (role, impl.pipeline_split_layer, split)
    rows[role] = {**actual, 'pipeline_split_layer': impl.pipeline_split_layer}
assert rows['actor'] == rows['ref']
assert not torch.cuda.is_initialized()
receipt = {'passed': True, 'typed_impl_configs': rows, 'model_constructed': False}
(source.parent / 'parallel-config.json').write_text(
    json.dumps(receipt, indent=2) + '\n'
)
print('DS41_TYPED_PARALLEL_CONFIG', json.dumps(receipt), flush=True)
