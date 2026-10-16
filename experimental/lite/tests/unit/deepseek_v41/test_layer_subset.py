# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Models built from renumbered release layers keep the release Engram hash."""
from functools import partial

import pytest
import torch
from megatron.lite.primitive.modules import engram_lookup as memory
from megatron.lite.primitive.modules.row_memory_build import build_row_memories
from test_redo_parity import release_config

ORDER, HEADS, VOCABULARY, COMPRESSED = 3, 2, 10, 7


def _build(layer_ids, hash_layer_ids=None):
    primes = memory.prime_buckets((1, 14), ORDER, HEADS, VOCABULARY)
    return build_row_memories(
        layer_ids=layer_ids,
        row_counts=primes.flatten(1).sum(1).tolist(),
        order=ORDER,
        heads=HEADS,
        vocabulary=VOCABULARY,
        width=32,
        compressed_vocabulary=COMPRESSED,
        pad_id=0,
        token_map=list(range(COMPRESSED)),
        token_count=COMPRESSED,
        hidden_size=32,
        copies=4,
        eps=1e-6,
        trainable=False,
        group=None,
        group_size=1,
        local_range=(0, 0),
        constructors=(
            memory.prime_buckets,
            memory.hash_multipliers,
            memory.NgramHash,
            memory.RowLookup,
            memory.EngramTable,
            memory.ShardedEngramTable,
            memory.Engram,
        ),
        projection=partial(torch.nn.Linear, bias=False),
        hash_layer_ids=hash_layer_ids,
    )[0]


def test_hash_defaults_to_table_layer_ids():
    expected = memory.hash_multipliers((1, 14), ORDER, COMPRESSED)
    torch.testing.assert_close(_build((1, 14)).multipliers, expected, rtol=0, atol=0)


def test_renumbered_tables_keep_release_hash():
    release = _build((1, 14))
    renumbered = _build((0, 1), hash_layer_ids=(1, 14))
    for name in ('multipliers', 'primes', 'offsets'):
        torch.testing.assert_close(
            getattr(renumbered, name), getattr(release, name), rtol=0, atol=0
        )
    assert not torch.equal(_build((0, 1)).multipliers, release.multipliers)


@pytest.mark.parametrize('hash_ids', [[14], [1, -14], [1, 14.0]])
def test_config_rejects_malformed_hash_layer_ids(hash_ids):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    release = release_config().to_hf_dict()
    release['text_config']['engram_hash_layer_ids'] = hash_ids
    with pytest.raises(ValueError, match='engram_hash_layer_ids'):
        DeepseekV41Config(release)


def test_config_round_trips_hash_layer_ids():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    release = release_config().to_hf_dict()
    release['text_config']['engram_hash_layer_ids'] = [1, 14]
    config = DeepseekV41Config(release)
    assert config.to_hf_dict()['text_config']['engram_hash_layer_ids'] == [1, 14]


def _layer_subset():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / 'examples/verl/scripts/ds41_w4a8_proxy/layer_subset.py'
    )
    spec = importlib.util.spec_from_file_location('ds41_layer_subset', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P6 = [1, 14, 15, 20, 21, 24]


def test_p6_config_keeps_every_release_role():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    release = release_config().to_hf_dict()
    text = _layer_subset().subset_config(release, P6)['text_config']
    assert text['num_hidden_layers'] == 6
    assert text['compress_ratios'] == [0, 2, 2, 1, 1, 1, 0, 0, 0]
    assert text['kv_source_layer_ids'] == [1, 3]
    assert text['index_source_layer_ids'] == [1, 3, 5]
    assert text['candidate_source_layer_id'] == 3
    assert text['engram_layer_ids'] == [0, 1]
    assert text['engram_num_embeddings'] == [18, 30]
    assert text['engram_hash_layer_ids'] == [1, 14]
    roles = [
        (p.kv_owner, p.index_owner, p.candidate_mode, p.engram_slot)
        for p in DeepseekV41Config({**release, 'text_config': text}).topology
    ]
    assert roles == [
        (None, None, 'none', 0),
        (1, 1, 'none', 1),
        (1, 1, 'none', None),
        (3, 3, 'build', None),
        (3, 3, 'reuse', None),
        (3, 5, 'reuse', None),
    ]


@pytest.mark.parametrize(
    'layers, message',
    [
        ([15, 20], 'needs release layers'),
        ([14, 15, 21], 'needs release layers'),
        ([0, 14], 'prefix of the release tables'),
        ([14, 1], 'unique and increasing'),
        ([1, 40], 'outside the release backbone'),
    ],
)
def test_subset_rejects_layers_without_their_release_roles(layers, message):
    release = release_config().to_hf_dict()
    with pytest.raises(ValueError, match=message):
        _layer_subset().subset_config(release, layers)


def test_renamed_shards_are_byte_exact_and_closed(tmp_path):
    from safetensors.torch import load_file, save_file

    module = _layer_subset()
    source = tmp_path / 'release'
    source.mkdir()
    tensors = {
        'embed.weight': torch.randn(4, 8),
        'layers.0.attn.wkv.weight': torch.randn(8, 8),
        'layers.1.engram.embed.weight': torch.randn(5, 8).to(torch.float8_e4m3fn),
        'layers.14.ffn.experts.0.w1.weight': torch.randint(0, 255, (3, 4)).to(
            torch.uint8
        ),
        'layers.2.attn.wkv.weight': torch.randn(8, 8),
        'mtp.0.attn.wkv.weight': torch.randn(2, 8),
    }
    weight_map = {}
    for index, names in enumerate(
        [
            [
                'embed.weight',
                'layers.0.attn.wkv.weight',
                'layers.1.engram.embed.weight',
            ],
            ['layers.14.ffn.experts.0.w1.weight', 'layers.2.attn.wkv.weight'],
            ['mtp.0.attn.wkv.weight'],
        ]
    ):
        file = f'model-{index}.safetensors'
        save_file({name: tensors[name] for name in names}, source / file)
        weight_map.update(dict.fromkeys(names, file))
    out = tmp_path / 'proxy'
    out.mkdir()

    new_map, lineage = module.write_subset(source, weight_map, [1, 14], out)

    expected = {
        'embed.weight': 'embed.weight',
        'layers.0.engram.embed.weight': 'layers.1.engram.embed.weight',
        'layers.1.ffn.experts.0.w1.weight': 'layers.14.ffn.experts.0.w1.weight',
        'mtp.0.attn.wkv.weight': 'mtp.0.attn.wkv.weight',
    }
    assert set(new_map) == set(expected)
    loaded = {}
    for file in set(new_map.values()):
        loaded.update(load_file(out / file))
    # No release layer is readable under a proxy index.
    assert set(loaded) == set(expected)
    for new, old in expected.items():
        assert loaded[new].dtype == tensors[old].dtype
        assert torch.equal(
            loaded[new].view(torch.uint8), tensors[old].view(torch.uint8)
        )
    assert {row['key']: row['source_key'] for row in lineage} == expected


def _audit(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    monkeypatch.setenv('W4_GRPO_OUT', str(tmp_path))
    path = (
        Path(__file__).resolve().parents[3]
        / 'examples/verl/scripts/ds41_w4a8_proxy/audit/ds41_audit.py'
    )
    spec = importlib.util.spec_from_file_location('ds41_audit_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _events(tmp_path):
    import json

    return [
        json.loads(line)
        for path in (tmp_path / 'audit').glob('events-*.jsonl')
        for line in path.read_text().splitlines()
    ]


@pytest.mark.parametrize('columns, dropped', [(8, 0), (24, 8)])
def test_training_candidate_audit_counts_dropped_positions(
    monkeypatch, tmp_path, columns, dropped
):
    from types import SimpleNamespace

    def candidate_blocks(scores, lengths, *, topk_blocks, block_size):
        # csa.candidate_blocks contract: a position mask of the kept blocks.
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[..., : topk_blocks * block_size] = True
        return keep

    audit = _audit(monkeypatch, tmp_path)
    owner = SimpleNamespace(candidate_blocks=candidate_blocks)
    audit._training_candidates(owner)
    scores = torch.zeros(1, 1, columns)

    keep = owner.candidate_blocks(
        scores, torch.tensor([[columns]]), topk_blocks=2, block_size=8
    )

    assert int(keep.sum()) == min(columns, 16)
    (row,) = [e for e in _events(tmp_path) if e['kind'] == 'candidate_blocks_call']
    assert row['owner'] == 'training' and row['dropped'] == dropped


def test_inference_candidate_audit_counts_dropped_blocks(monkeypatch, tmp_path):
    from types import SimpleNamespace

    def select_candidate_blocks(
        logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat=1
    ):
        # vLLM's contract: block ids by max score, newest pinned, -1 padded.
        blocks = (logits.shape[1] + block_size - 1) // block_size
        out.fill_(-1)
        out[:, : min(topk_blocks, blocks)] = torch.arange(min(topk_blocks, blocks))

    audit = _audit(monkeypatch, tmp_path)
    owner = SimpleNamespace(_select_candidate_blocks=select_candidate_blocks)
    audit._inference_candidates(owner)
    out = torch.empty(2, 2, dtype=torch.int32)
    # Row 0 reaches 3 blocks, row 1 reaches 1: one block dropped in total.
    owner._select_candidate_blocks(
        torch.zeros(2, 24), None, torch.tensor([24, 8]), 2, 8, out
    )

    (row,) = [e for e in _events(tmp_path) if e['kind'] == 'candidate_blocks_call']
    assert row['owner'] == 'inference' and row['dropped'] == 1


def test_first_layer_broadcast_only_without_engram():
    from megatron.lite.model.deepseek_v41 import topology

    release = topology.V41_TOPOLOGY
    assert release[0].broadcasts_embedding
    assert not any(policy.broadcasts_embedding for policy in release[1:])

    text = _layer_subset().subset_config(release_config().to_hf_dict(), P6)[
        'text_config'
    ]
    proxy = topology.build_topology(_layer_subset()._spec(text))
    # Layer 0 is release layer 1: Engram injects first, so the copies differ.
    assert proxy[0].engram_slot == 0 and not proxy[0].broadcasts_embedding
