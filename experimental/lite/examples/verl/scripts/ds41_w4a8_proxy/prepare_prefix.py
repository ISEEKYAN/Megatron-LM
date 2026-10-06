"""Faithful release 2L closed prefix: read-only shard links, exact tokenizer map."""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from token_map import build_compressed_token_map
from transformers import AutoTokenizer, GenerationConfig

source = Path(os.environ['DS41_RELEASE'])
out = Path(os.environ['DS41_OUTPUT'])
model = Path(os.environ['DS41_MODEL'])
model.mkdir(parents=True, exist_ok=True)
receipts = {
    v['file']: v
    for p in source.glob('*.verified.json')
    if (v := json.loads(p.read_text()))
}
revision = 'dba1be0a40aa45a94ad051997016db3960a90277'
for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
    rec = receipts[name]
    assert rec['revision'] == revision
    actual = hashlib.file_digest((source / name).open('rb'), 'sha256').hexdigest()
    assert actual == rec['sha256'], name
cfg = json.loads((source / 'config.json').read_text())
original = json.loads(json.dumps(cfg))
t = cfg['text_config']
assert (
    t['hidden_size'],
    t['hc_mult'],
    t['n_routed_experts'],
    t['moe_intermediate_size'],
    t['num_experts_per_tok'],
) == (5120, 4, 384, 2304, 6)
assert t['compress_ratios'][:2] == [0, 0] and t['engram_layer_ids'][0] == 1
# A closed text prefix: CR0/0 has no outside owner or borrowed state; retain
# the 3 archival MTP metadata slots and all release matrix dimensions.
t.update(
    num_hidden_layers=2,
    compress_ratios=original['text_config']['compress_ratios'][:2]
    + original['text_config']['compress_ratios'][-3:],
    candidate_source_layer_id=-1,
    kv_source_layer_ids=[],
    index_source_layer_ids=[],
    engram_layer_ids=[1],
    engram_num_embeddings=[original['text_config']['engram_num_embeddings'][0]],
)
allowed = {
    'num_hidden_layers',
    'compress_ratios',
    'candidate_source_layer_id',
    'kv_source_layer_ids',
    'index_source_layer_ids',
    'engram_layer_ids',
    'engram_num_embeddings',
}
assert {k for k in t if t[k] != original['text_config'][k]} <= allowed
# Existing VERL generic-HF adapter reads this root metadata alias. Its value
# is identical to the release's nested context limit; no resizing/scaling.
cfg['max_position_embeddings'] = t['max_position_embeddings']
(model / 'config.json').write_text(json.dumps(cfg, indent=2))
for name in ('tokenizer.json', 'tokenizer_config.json'):
    shutil.copyfile(source / name, model / name)
tok = AutoTokenizer.from_pretrained(
    model, trust_remote_code=True, local_files_only=True
)
lookup, size = build_compressed_token_map(tok)
assert (
    len(tok) == t['vocab_size'] == 129280
    and size == t['engram_compressed_vocab_size'] == 99092
), (len(tok), size)
(model / 'token-map.json').write_text(json.dumps(lookup))
# Chat rendering is a declared experiment input. Keep tokenizer JSON/config
# byte-identical; use the existing VERL custom_chat_template config separately.
index = json.loads((source / 'model.safetensors.index.json').read_text())
weights = {}
for key, file in index['weight_map'].items():
    m = re.match(r'layers\.(\d+)\.', key)
    if m is None or int(m.group(1)) < 2:
        weights[key] = file
files = sorted(set(weights.values()))
lineage = []
for file in files:
    rec = receipts[file]
    assert (
        rec['revision'] == revision
        and rec['lfs_verified']
        and rec['sha256'] == rec['expected_lfs_sha256']
    ), file
    assert (source / file).stat().st_size == rec['size']
    dest = model / file
    if not dest.exists():
        dest.symlink_to(source / file)
    lineage.append(
        {
            'file': file,
            **rec,
            'validation': 'existing exact-revision verified LFS receipt + present size; no duplicate 100GB rehash',
        }
    )
assert len(files) == 9 and any('00047' in f for f in files), files
(model / 'model.safetensors.index.json').write_text(
    json.dumps(
        {
            'metadata': {'prefix_layers': [0, 1], 'release_revision': revision},
            'weight_map': weights,
        },
        indent=2,
    )
)
GenerationConfig(
    bos_token_id=cfg['bos_token_id'],
    eos_token_id=cfg['eos_token_id'],
    pad_token_id=cfg['pad_token_id'],
).save_pretrained(model)
data = Path(os.environ['DS41_DATA'])
data_sha = hashlib.file_digest(data.open('rb'), 'sha256').hexdigest()
assert data_sha == 'a2c2e65314f40219e8830a60c01cbbc848982a48b274e78083c50e702d29adbd'
import pyarrow.parquet as pq

assert pq.ParquetFile(data).metadata.num_rows == 17391
record = {
    'release_revision': revision,
    'source_config_sha256': receipts['config.json']['sha256'],
    'prefix_config': cfg,
    'changed_text_fields': sorted(k for k in t if t[k] != original['text_config'][k]),
    'weights': lineage,
    'active_and_archival_index_keys': len(weights),
    'tokenizer_sha256': receipts['tokenizer.json']['sha256'],
    'token_map_sha256': hashlib.file_digest(
        (model / 'token-map.json').open('rb'), 'sha256'
    ).hexdigest(),
    'compressed_vocabulary': size,
    'training_sha': os.environ.get('W4_TRAIN_SHA', 'caller environment'),
    'vllm_sha': os.environ.get('W4_VLLM_SHA', 'caller environment'),
    'data_sha256': data_sha,
    'reward': 'explicit pipeline proxy: numeric non-whitespace character fraction (same tiny function); unmodified DAPO correctness computed and recorded independently, not training reward; secretary authorization 2026-10-05',
    'training_scope': 'all 384 experts/layer active, original widths and optimizer; published default frozen resident FP8 Engram table, trainable projection/norm unchanged; no extra parameter freezing',
}
(out / 'input-lineage.json').write_text(json.dumps(record, indent=2))
print(
    'REAL_PREFIX_LINEAGE_PASS '
    + json.dumps(
        {
            k: record[k]
            for k in [
                'release_revision',
                'changed_text_fields',
                'active_and_archival_index_keys',
                'compressed_vocabulary',
                'token_map_sha256',
            ]
        }
    ),
    flush=True,
)
