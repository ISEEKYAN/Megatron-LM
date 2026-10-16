"""Release layer-subset proxy: renumbered byte-exact shards, release roles and hash."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from layer_subset import subset_config, write_subset
from token_map import build_compressed_token_map
from transformers import AutoTokenizer, GenerationConfig

source = Path(os.environ['DS41_RELEASE'])
out = Path(os.environ['DS41_OUTPUT'])
model = Path(os.environ['DS41_MODEL'])
layers = [int(i) for i in os.environ['DS41_VARIANT_LAYERS'].split(',')]
# Same precision switch as prepare_prefix.py: official adds the reference
# inference contract (group-32 expert activations, BF16 wo_a) to the config.
precision = os.environ.get('DS41_PRECISION', 'release')
if precision not in ('release', 'official'):
    raise ValueError('DS41_PRECISION must be release or official')
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
cfg = subset_config(json.loads((source / 'config.json').read_text()), layers)
t = cfg['text_config']
if precision == 'official':
    t.update(expert_activation_group_size=32, wo_a_dtype='bf16')
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
index = json.loads((source / 'model.safetensors.index.json').read_text())
for file in sorted(set(index['weight_map'].values())):
    rec = receipts[file]
    assert (
        rec['revision'] == revision
        and rec['lfs_verified']
        and rec['sha256'] == rec['expected_lfs_sha256']
    ), file
    assert (source / file).stat().st_size == rec['size']
weights, lineage = write_subset(source, index['weight_map'], layers, model)
(model / 'model.safetensors.index.json').write_text(
    json.dumps(
        {
            'metadata': {'release_layers': layers, 'release_revision': revision},
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
release_text = json.loads((source / 'config.json').read_text())['text_config']
record = {
    'release_revision': revision,
    'release_layers': layers,
    'source_config_sha256': receipts['config.json']['sha256'],
    'prefix_config': cfg,
    'changed_text_fields': sorted(
        k for k in t if k not in release_text or t[k] != release_text[k]
    ),
    'tensors': lineage,
    'tokenizer_sha256': receipts['tokenizer.json']['sha256'],
    'token_map_sha256': hashlib.file_digest(
        (model / 'token-map.json').open('rb'), 'sha256'
    ).hexdigest(),
    'compressed_vocabulary': size,
    'training_sha': os.environ.get('W4_TRAIN_SHA', 'caller environment'),
    'vllm_sha': os.environ.get('W4_VLLM_SHA', 'caller environment'),
    'data_sha256': data_sha,
}
(out / 'input-lineage.json').write_text(json.dumps(record, indent=2))
print(
    'REAL_SUBSET_LINEAGE_PASS '
    + json.dumps(
        {
            'release_revision': revision,
            'release_layers': layers,
            'changed_text_fields': record['changed_text_fields'],
            'tensors': len(lineage),
            'token_map_sha256': record['token_map_sha256'],
        }
    ),
    flush=True,
)
