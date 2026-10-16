"""Renumber a closed subset of release decoder layers into a proxy checkpoint.

Each kept layer keeps its release role: its KV/index owners and candidate
publisher must be kept too, Engram tables keep their slot order (and with it
their prime buckets) and their release hash seed via `engram_hash_layer_ids`.
Tensors are copied byte for byte into fresh shards under renumbered names, so
no release layer can be read under a proxy layer index.
"""

import hashlib
import json
import re
import struct
from copy import deepcopy
from pathlib import Path

from megatron.lite.model.deepseek_v41.topology import TopologySpec, build_topology

LAYER_KEY = re.compile(r'layers\.(\d+)\.')
COPY_CHUNK = 1 << 26


def _spec(text):
    return TopologySpec(
        **{
            k: tuple(text[k]) if isinstance(text[k], list) else text[k]
            for k in TopologySpec.__dataclass_fields__
        }
    )


def subset_config(release, layers):
    """Release HF config -> proxy config over `layers` (release ids, sorted)."""
    layers = list(layers)
    if not layers or layers != sorted(set(layers)):
        raise ValueError('Layers must be unique and increasing')
    text = release['text_config']
    count = text['num_hidden_layers']
    if layers[-1] >= count:
        raise ValueError('Layer outside the release backbone')
    policies = build_topology(_spec(text))
    new = {layer: index for index, layer in enumerate(layers)}
    candidate = text['candidate_source_layer_id']
    for layer in layers:
        policy = policies[layer]
        owners = [policy.kv_owner, policy.index_owner] if policy.compress_ratio else []
        if policy.candidate_mode != 'none':
            owners.append(candidate)
        missing = [owner for owner in owners if owner not in new]
        if missing:
            raise ValueError(f'Layer {layer} needs release layers {missing}')
    engram = [layer for layer in layers if layer in text['engram_layer_ids']]
    if engram != list(text['engram_layer_ids'][: len(engram)]):
        raise ValueError('Kept Engram tables must be a prefix of the release tables')
    ratios = text['compress_ratios']
    out = deepcopy(release)
    out['text_config'].update(
        num_hidden_layers=len(layers),
        compress_ratios=[ratios[layer] for layer in layers] + ratios[count:],
        kv_source_layer_ids=[new[i] for i in text['kv_source_layer_ids'] if i in new],
        index_source_layer_ids=[
            new[i] for i in text['index_source_layer_ids'] if i in new
        ],
        candidate_source_layer_id=new.get(candidate, -1),
        engram_layer_ids=[new[layer] for layer in engram],
        engram_num_embeddings=list(text['engram_num_embeddings'][: len(engram)]),
        engram_hash_layer_ids=engram,
    )
    # The renumbered graph must reproduce every kept layer's release role.
    for layer, policy in zip(layers, build_topology(_spec(out['text_config']))):
        release_policy = policies[layer]
        expected = [
            release_policy.compress_ratio,
            *(
                None if owner is None else new[owner]
                for owner in (release_policy.kv_owner, release_policy.index_owner)
            ),
            release_policy.candidate_mode,
            release_policy.engram_rows,
            release_policy.engram_slot,
        ]
        actual = [
            policy.compress_ratio,
            policy.kv_owner,
            policy.index_owner,
            policy.candidate_mode,
            policy.engram_rows,
            policy.engram_slot,
        ]
        if actual != expected:
            raise ValueError(f'Layer {layer} changes role: {actual} != {expected}')
    return out


def renamed(key, layers):
    """Release tensor name -> proxy name, or None for a dropped layer."""
    match = LAYER_KEY.match(key)
    if match is None:
        return key
    layer = int(match.group(1))
    if layer not in layers:
        return None
    return f'layers.{layers.index(layer)}.' + key[match.end() :]


def _header(path):
    with open(path, 'rb') as f:
        (size,) = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(size))
    return header, 8 + size


def copy_shard(source, names, destination):
    """Copy `names` ({new: old}) from one source shard; return per-tensor sha256."""
    header, base = _header(source)
    entries, offset = {}, 0
    for new, old in names.items():
        info = header[old]
        begin, end = info['data_offsets']
        entries[new] = dict(
            dtype=info['dtype'],
            shape=info['shape'],
            data_offsets=[offset, offset + end - begin],
        )
        offset += end - begin
    body = json.dumps({'__metadata__': {'format': 'pt'}, **entries}).encode()
    body += b' ' * (-len(body) % 8)
    digests = {}
    with open(source, 'rb') as src, open(destination, 'wb') as dst:
        dst.write(struct.pack('<Q', len(body)) + body)
        for new, old in names.items():
            begin, end = header[old]['data_offsets']
            src.seek(base + begin)
            digest, remaining = hashlib.sha256(), end - begin
            while remaining:
                chunk = src.read(min(COPY_CHUNK, remaining))
                if not chunk:
                    raise IOError(f'{source} ended inside {old}')
                digest.update(chunk)
                dst.write(chunk)
                remaining -= len(chunk)
            digests[new] = digest.hexdigest()
    return digests


def tensor_digests(path):
    """sha256 of each tensor's bytes, read back from a written shard."""
    header, base = _header(path)
    digests = {}
    with open(path, 'rb') as f:
        for name, info in header.items():
            if name == '__metadata__':
                continue
            begin, end = info['data_offsets']
            f.seek(base + begin)
            digest, remaining = hashlib.sha256(), end - begin
            while remaining:
                chunk = f.read(min(COPY_CHUNK, remaining))
                digest.update(chunk)
                remaining -= len(chunk)
            digests[name] = digest.hexdigest()
    return digests


def _plan(weight_map, layers):
    plan = {}
    for old, file in weight_map.items():
        new = renamed(old, layers)
        if new is not None:
            plan.setdefault(file, {})[new] = old
    return plan


def _shard_name(index, count):
    return f'model-subset-{index + 1:05d}-of-{count:05d}.safetensors'


def source_digests(source, names):
    """sha256 of each release tensor `names` ({new: old}) read from `source`."""
    header, base = _header(source)
    digests = {}
    with open(source, 'rb') as f:
        for new, old in names.items():
            begin, end = header[old]['data_offsets']
            f.seek(base + begin)
            digest, remaining = hashlib.sha256(), end - begin
            while remaining:
                chunk = f.read(min(COPY_CHUNK, remaining))
                if not chunk:
                    raise IOError(f'{source} ended inside {old}')
                digest.update(chunk)
                remaining -= len(chunk)
            digests[new] = digest.hexdigest()
    return digests


def reuse_subset(source_dir, weight_map, layers, prepared_dir, output_dir):
    """Link the shards of an earlier `write_subset` of the same layers.

    Every reused tensor is read back and must equal the release bytes under its
    release name (same dtype, shape and sha256) before it is linked; nothing is
    copied. Returns the same weight map and lineage as `write_subset`.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    prepared_dir = Path(prepared_dir).resolve()
    plan = _plan(weight_map, layers)
    new_map, lineage = {}, []
    for index, file in enumerate(sorted(plan)):
        name = _shard_name(index, len(plan))
        header, _ = _header(source_dir / file)
        reused, _ = _header(prepared_dir / name)
        reused.pop('__metadata__', None)
        if set(reused) != set(plan[file]):
            raise ValueError(f'{name} does not hold the planned tensors')
        for new, old in plan[file].items():
            if [reused[new]['dtype'], reused[new]['shape']] != [
                header[old]['dtype'],
                header[old]['shape'],
            ]:
                raise ValueError(f'{name}:{new} differs from release {old}')
        expected = source_digests(source_dir / file, plan[file])
        if tensor_digests(prepared_dir / name) != expected:
            raise IOError(f'{name} does not hold the release bytes')
        target = output_dir / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(prepared_dir / name)
        for new, old in plan[file].items():
            new_map[new] = name
            lineage.append(
                dict(key=new, source_key=old, source_file=file, sha256=expected[new])
            )
    return new_map, lineage


def write_subset(source_dir, weight_map, layers, output_dir):
    """Write renumbered shards for `layers`; return the new weight map and lineage."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    plan = _plan(weight_map, layers)
    new_map, lineage = {}, []
    for index, file in enumerate(sorted(plan)):
        name = _shard_name(index, len(plan))
        copied = copy_shard(source_dir / file, plan[file], output_dir / name)
        written = tensor_digests(output_dir / name)
        if written != copied:
            raise IOError(f'{name} does not read back the copied bytes')
        for new, old in plan[file].items():
            new_map[new] = name
            lineage.append(
                dict(key=new, source_key=old, source_file=file, sha256=copied[new])
            )
    return new_map, lineage
