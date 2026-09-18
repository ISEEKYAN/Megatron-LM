# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import ast
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

# Reference source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277
# Fetched at 2026-09-17T17:25:23.437086+00:00; all seven files matched this revision.
# Per-file hashes below verify integrity; source provenance is the repo@commit.
REFERENCE_SHA256 = {
    'model.py': '4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65',
    'kernel.py': '1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455',
    'engram.py': '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897',
    'vision.py': '5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c',
    'image_processor.py': '482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272',
    'inference_config.json': '2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809',
    'config.json': '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879',
}


@pytest.fixture
def official(monkeypatch):
    root = Path(os.environ['DS41_REFERENCE_DIR'])
    for name, digest in REFERENCE_SHA256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

    def load(name, cls=None, method=None):
        path = root / name
        if cls:

            def named(nodes, kind, label):
                return next(n for n in nodes if isinstance(n, kind) and n.name == label)

            node = named(ast.parse(path.read_text()).body, ast.ClassDef, cls)
            node = named(node.body, ast.FunctionDef, method)
            namespace = {'torch': torch, 'nn': torch.nn}
            tree = ast.Module(body=[node], type_ignores=[])
            exec(compile(tree, str(path), 'exec'), namespace)
            return namespace[method]
        spec = importlib.util.spec_from_file_location('official_' + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    return load
