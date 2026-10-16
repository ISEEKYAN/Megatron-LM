# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Full HF checkpoint loads into stage-local bindings without dropping guards."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def stage_checkpoint(v41_core_te, tmp_path, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import _SPEC
    from megatron.lite.primitive.ckpt.binding_records import TensorBinding

    names = ['embed.weight', 'layers.0.weight', 'layers.2.weight', 'head.weight']
    tensors = {name: torch.full((2, 2), float(i + 1)) for i, name in enumerate(names)}
    save_file(tensors, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'config.json').write_text('{}')
    groups = [set(names[:2]), set(names[2:])]
    models = []
    for rank in (0, 1):
        bindings = {}
        for name in sorted(groups[rank]):
            owner = torch.nn.Linear(2, 2, bias=False)
            owner.weight.data.zero_()
            bindings[name] = TensorBinding(name, owner, 'weight', 'parameter')
        models.append(
            SimpleNamespace(
                ps=SimpleNamespace(
                    pp_size=2, pp_rank=rank, pp_group=object(), ep_size=1
                ),
                tensor_bindings=bindings,
                archival_bindings={},
                config=SimpleNamespace(to_hf_dict=lambda: {}),
            )
        )

    def gather(out, value, *, group):
        assert group in [m.ps.pp_group for m in models]
        out[:] = groups

    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    return models, _SPEC, tmp_path, tensors


@pytest.mark.parametrize('stage', [0, 1])
def test_full_hf_payload_loads_only_owned_stage(stage_checkpoint, stage):
    from megatron.lite.primitive.ckpt.hf_weights import load_bound_model

    models, spec, root, tensors = stage_checkpoint
    model = models[stage]
    load_bound_model(model, root, spec)
    for name, binding in model.tensor_bindings.items():
        assert torch.equal(binding.tensor, tensors[name])


@pytest.mark.parametrize('fault', ['missing_local', 'unknown_remote', 'unknown_root'])
@pytest.mark.parametrize('stage', [0, 1])
def test_pp_header_coverage_fails_closed(stage_checkpoint, stage, fault):
    from megatron.lite.primitive.ckpt.hf_weights import validate_bound_checkpoint_keys

    models, spec, root, tensors = stage_checkpoint
    model = models[stage]
    keys = set(tensors)
    if fault == 'missing_local':
        keys.remove(next(iter(model.tensor_bindings)))
    elif fault == 'unknown_remote':
        keys.add(f'layers.{2 if stage == 0 else 0}.typo.weight')
    else:
        keys.add('unexpected.weight')
    with pytest.raises(ValueError, match='Checkpoint key coverage mismatch'):
        validate_bound_checkpoint_keys(model, keys, spec)


@pytest.mark.parametrize('stage', [0, 1])
def test_pp_training_masters_are_scoped_too(stage_checkpoint, stage):
    from megatron.lite.primitive.ckpt.hf_weights import load_bound_model

    models, spec, root, tensors = stage_checkpoint
    masters = root / 'mlite_masters'
    masters.mkdir()
    tensors = {name: value + 10 for name, value in tensors.items()}
    save_file(tensors, str(masters / 'model.safetensors'))
    load_bound_model(models[stage], root, spec)
    for name, binding in models[stage].tensor_bindings.items():
        assert torch.equal(binding.tensor, tensors[name])


def test_pp1_still_rejects_unbound_keys(stage_checkpoint):
    from megatron.lite.primitive.ckpt.hf_weights import validate_bound_checkpoint_keys

    models, spec, root, tensors = stage_checkpoint
    models[0].ps.pp_size = 1
    with pytest.raises(ValueError, match='Checkpoint key coverage mismatch'):
        validate_bound_checkpoint_keys(models[0], set(tensors), spec)
