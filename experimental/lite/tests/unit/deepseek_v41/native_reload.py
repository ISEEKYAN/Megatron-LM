# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run installed upstream CPU functions without importing CUDA-only packages.

DS41_NATIVE_SOURCE_ROOT may point to a read-only site-packages snapshot. We
compile the original function bodies, with deferred annotations, rather than
copying layout equations into test doubles. No production source is modified.
"""
import ast
import importlib.machinery
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def source(relative):
    root = os.environ.get('DS41_NATIVE_SOURCE_ROOT')
    if root:
        path = Path(root) / relative
        assert path.is_file(), f'Missing native source: {path}'
        return path
    package = relative.split('/')[0]
    # Lifecycle tests replace package objects in sys.modules; resolve the
    # installed source on sys.path without consulting those test doubles.
    spec = importlib.machinery.PathFinder.find_spec(package)
    if spec is None:
        pytest.skip(f'{package} source required for native CPU layout test')
    return Path(next(iter(spec.submodule_search_locations))).parent / relative


def functions(relative, names, class_name=None, **bindings):
    path = source(relative)
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(
            n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    selected = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            selected.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in names
        ):
            selected.append(node)
    assert len(selected) == len(names), (path, names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module='__future__', names=[ast.alias(name='annotations')], level=0
            )
        ]
        + selected,
        type_ignores=[],
    )
    namespace = {'torch': torch, **bindings}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def mxfp4_method():
    def set_attrs(param, attrs):
        for name, value in attrs.items():
            setattr(param, name, value)

    native = functions(
        'vllm/model_executor/layers/quantization/mxfp4.py',
        ['create_weights', 'get_scale_weight_loader', '_encode_mxfp4_weight_scale'],
        'Mxfp4MoEMethod',
        set_weight_attrs=set_attrs,
    )
    method_type = type(
        'Mxfp4MoEMethod',
        (),
        {
            name: getattr(native, name)
            for name in (
                'create_weights',
                'get_scale_weight_loader',
                '_encode_mxfp4_weight_scale',
            )
        },
    )
    native.create_weights.__globals__['Mxfp4MoEMethod'] = method_type
    method = method_type()
    method.moe = SimpleNamespace(w13_num_shards=2, has_bias=False)
    method._cache_permute_indices = {}
    return method


def fused_layout():
    path = source('vllm/models/deepseek_v41/common/ops/fused_layout.py')
    spec = importlib.util.spec_from_file_location('native_fused_layout', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def permute_builders():
    utils = functions(
        'flashinfer/utils.py',
        [
            'srcToDstBlk16RowMap',
            'srcToDstBlk32RowMap',
            'get_shuffle_block_size',
            'get_shuffle_matrix_a_row_indices',
            'get_shuffle_matrix_sf_a_row_indices',
        ],
    )
    return functions(
        'flashinfer/fused_moe/core.py',
        [
            '_maybe_get_cached_w3_w1_permute_indices',
            'get_w2_permute_indices_with_cache',
            'get_reorder_rows_for_gated_act_gemm_row_indices',
        ],
        **vars(utils),
    )


def online_loader(info, process, monkeypatch):
    import inspect
    from functools import wraps

    from torch.utils._python_dispatch import TorchDispatchMode

    base = 'vllm/model_executor/model_loader/reload/'
    utils = functions(
        base + 'utils.py',
        [
            'get_layer_tensors',
            'get_layer_params_buffers',
            'get_tensor_load_numel',
            'get_layer_size',
            'has_device_tensors',
        ],
    )
    meta = functions(
        base + 'meta.py',
        ['CopyCounter', 'get_numel_loaded', 'SKIP_LOAD_TENSORS'],
        TorchDispatchMode=TorchDispatchMode,
        get_tensor_load_numel=utils.get_tensor_load_numel,
    )
    # Keep the native get_layer_size relative import intact.
    import sys
    from types import ModuleType

    relative_meta = ModuleType('vllm.model_executor.model_loader.reload.meta')
    relative_meta.SKIP_LOAD_TENSORS = meta.SKIP_LOAD_TENSORS
    monkeypatch.setitem(sys.modules, relative_meta.__name__, relative_meta)
    utils.get_layer_size.__globals__['__package__'] = (
        'vllm.model_executor.model_loader.reload'
    )
    bindings = dict(vars(utils))
    bindings.update(
        inspect=inspect,
        wraps=wraps,
        get_numel_loaded=meta.get_numel_loaded,
        get_layerwise_info=lambda layer: info,
        default_weight_loader=None,
        _wrap_parameters_weight_loader=lambda layer: None,
        is_deferred_attention_layer=lambda layer: False,
        LOADING_LAYERS=set(),
        _layerwise_process=process,
        logger=SimpleNamespace(debug=lambda *a: None),
    )
    native = functions(
        base + 'layerwise.py',
        ['make_online_process_loader', '_get_original_loader', '_get_weight_loader'],
        **bindings,
    )
    return native.make_online_process_loader
