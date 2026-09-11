"""Pinned original methods for CPU floating diagnostics, not quantized parity.

Only GPU primitives are substituted: quantizers are explicitly disabled and
sparse attention uses the scalar gather/softmax definition below. Source methods
are otherwise unchanged. This profile cannot certify B1 native GPU execution.
"""

import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from fixtures import REFERENCE_SHA256
from torch import nn
from torch.nn import functional as F


def validate_cpu_source(directory, name="model.py"):
    if (
        hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
        != REFERENCE_SHA256[name]
    ):
        raise ValueError(f"Pinned CPU reference digest mismatch: {name}")


def sparse_attention_cpu(q, kv, sink, indices, scale):
    batches = []
    for b in range(q.shape[0]):
        positions = []
        for pos in range(q.shape[1]):
            keys = kv[b, indices[b, pos][indices[b, pos] >= 0].long()]
            heads = []
            for h in range(q.shape[2]):
                logits = (keys.float() @ q[b, pos, h].float()) * scale
                probs = torch.cat([logits, sink[h : h + 1].float()]).softmax(0)[:-1]
                heads.append((probs[:, None] * keys.float()).sum(0))
            positions.append(torch.stack(heads))
        batches.append(torch.stack(positions))
    return torch.stack(batches).to(q.dtype)


def load_floating_reference(directory):
    directory = Path(directory)
    validate_cpu_source(directory)
    names = {
        'ModelArgs',
        'Linear',
        'ColumnParallelLinear',
        'RowParallelLinear',
        'RMSNorm',
        'Compressor',
        'Indexer',
        'Attention',
        'Gate',
        'Expert',
        'linear',
        'precompute_freqs_cis',
        'apply_rotary_emb',
        'get_window_topk_idxs',
        'select_candidate_blocks',
    }
    tree = ast.parse((directory / 'model.py').read_text())
    nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
    assert {node.name for node in nodes} == names
    # Decorators on cached position functions only memoize values. Keep the
    # original functions; import lru_cache rather than modifying their bodies.
    from dataclasses import dataclass
    from functools import lru_cache
    from typing import Literal

    namespace = dict(
        torch=torch,
        nn=nn,
        F=F,
        math=math,
        dataclass=dataclass,
        Literal=Literal,
        lru_cache=lru_cache,
        world_size=1,
        rank=0,
        default_dtype=torch.float32,
        fp8_block_size=32,
        fp4_block_size=32,
        scale_fmt='ue8m0',
        scale_dtype=torch.float8_e8m0fnu,
        shared_attn=SimpleNamespace(),
        sparse_attn=sparse_attention_cpu,
        act_quant=lambda *args, **kwargs: None,
        fp4_act_quant=lambda *args, **kwargs: None,
    )
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(directory / 'model.py'), 'exec'
        ),
        namespace,
    )
    return namespace


def original_method(directory, class_name, method_name):
    directory = Path(directory)
    validate_cpu_source(directory)
    tree = ast.parse((directory / 'model.py').read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    namespace = dict(torch=torch, F=F)
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(directory / 'model.py'),
            'exec',
        ),
        namespace,
    )
    return namespace[method_name]
