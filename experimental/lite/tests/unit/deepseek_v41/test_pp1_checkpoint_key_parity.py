# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PP1 retains the pre-refactor key contract on real official subset indexes.

Fixtures contain the actual 4L/6L/9L configs and index weight_maps (no payload).
The oracle below is copied from load_bound_model at e324f2fdb7, before the
coverage extraction, rather than calling the new production helper twice.
"""
import gzip
import json
from pathlib import Path

import pytest
import torch


def _old_load_key_gate(model, keys, spec, *, allow_missing_archive=False):
    bindings = spec.expand_bindings(
        model, {**model.tensor_bindings, **model.archival_bindings}
    )
    active = {
        name for name, b in bindings.items() if b.role not in ('scale', 'archival')
    }
    required_archive = set(model.archival_bindings)
    if allow_missing_archive and not any(
        name.startswith(spec.optional_prefix) for name in keys
    ):
        required_archive = set()
    expected = active | required_archive
    if not expected <= keys or keys - set(bindings):
        raise ValueError('Checkpoint key coverage mismatch')
    return active, required_archive


def _outcome(gate, model, keys, spec, allow_missing_archive):
    try:
        return (
            'accepted',
            gate(model, keys, spec, allow_missing_archive=allow_missing_archive),
        )
    except ValueError as error:
        return ('rejected', str(error))


@pytest.mark.parametrize('variant', ['4l', '6l', '9l'])
def test_pp1_real_headers_match_pre_refactor_gate(v41_core_te, monkeypatch, variant):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite.checkpoint import _SPEC
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model
    from megatron.lite.primitive.ckpt.hf_weights import validate_bound_checkpoint_keys
    from megatron.lite.primitive.parallel.state import ParallelState

    path = Path(__file__).parent / 'fixtures/pp1_release_indexes' / f'{variant}.json.gz'
    fixture = json.loads(gzip.decompress(path.read_bytes()))
    keys = set(fixture['index']['weight_map'])
    assert len(keys) == {'4l': 12020, '6l': 16704, '9l': 23714}[variant]
    assert fixture['config']['text_config']['expert_activation_group_size'] == 32
    assert fixture['config']['text_config']['wo_a_dtype'] == 'bf16'
    with torch.device('meta'):
        model = DeepseekV41Model(
            DeepseekV41Config(fixture['config']),
            quantized=True,
            w4a8_experts=True,
            deployment_math=True,
            shard_engram=False,
            parallel_state=ParallelState(ep_size=8, ep_rank=0),
        )
    assert model.ps.pp_size == 1

    def forbidden(*args, **kwargs):
        raise AssertionError('PP1 key validation must not communicate')

    monkeypatch.setattr(torch.distributed, 'all_gather_object', forbidden)
    bindings = _SPEC.expand_bindings(
        model, {**model.tensor_bindings, **model.archival_bindings}
    )
    assert _SPEC.checkpoint_keys(model, keys, bindings) is keys
    active, archive = _old_load_key_gate(model, keys, _SPEC)
    assert active and archive
    missing = min(active)
    archive_missing = min(archive)
    no_mtp = {key for key in keys if not key.startswith(_SPEC.optional_prefix)}
    cases = [
        ('valid', keys, False, True),
        ('missing_weight', keys - {missing}, False, False),
        ('missing_archive', keys - {archive_missing}, False, False),
        ('unknown_weight', keys | {'layers.0.invalid.weight'}, False, False),
        ('unknown_scale', keys | {'layers.0.invalid.scale'}, False, False),
        ('missing_mtp_strict', no_mtp, False, False),
        ('missing_mtp_allowed', no_mtp, True, True),
    ]
    for label, candidate, allow, accepted in cases:
        original_keys = candidate.copy()
        old = _outcome(_old_load_key_gate, model, candidate, _SPEC, allow)
        new = _outcome(validate_bound_checkpoint_keys, model, candidate, _SPEC, allow)
        assert old == new, (variant, label, old, new)
        assert (new[0] == 'accepted') is accepted, (variant, label)
        assert candidate == original_keys
    # The master scope is the identical PP1 early return; subset rejection
    # remains master_keys - active, including the unexpected-master negative.
    for master_keys in [set(), {min(active)}, set(active), {'invalid.master.weight'}]:
        scoped = _SPEC.checkpoint_keys(model, master_keys, dict.fromkeys(active))
        assert scoped is master_keys
        assert (scoped - active) == (master_keys - active)
    print(f'PP1_HEADER_PARITY {variant} keys={len(keys)} cases={len(cases)} masters=4')
