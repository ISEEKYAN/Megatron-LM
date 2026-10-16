# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Closed set of native entry points on the per-document CP path.

``scan_native_entries`` enumerates, from the MLite source, every native entry
point MLite can reach: ``torch.ops.<ns>.<op>`` outside aten/prim, and every
callable taken from a vLLM, DeepGEMM or MLite-kernel module (``from m import
f``, ``import m as a`` / ``a = importlib.import_module('m')`` then ``a.f``).

``NativeCallTracker`` wraps those callables and watches the dispatcher while
the model runs. Every native call is attributed to the CP state of the current
document on this rank (``zero``/``one``/``part``/``whole`` local rows). It fails
loud on a native call that the scan did not enumerate, and on any native entry
handed an empty tensor unless that entry is listed in ``EMPTY_SAFE`` with its
evidence.
"""

import ast
import contextlib
import dataclasses
import functools
import importlib
import inspect
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

NATIVE_PREFIXES = ('vllm', 'deep_gemm', 'megatron.lite.primitive.kernels')
# DeepGEMM is resolved at run time (w4a8_experts._import_deep_gemm) in vLLM's order.
DYNAMIC_MODULES = {
    'deep_gemm': ('deep_gemm', 'vllm.third_party.deep_gemm'),
    # ``importlib.import_module(name)`` with a computed name (w4a8_experts resolves
    # DeepGEMM this way); its attributes are resolved against the same modules.
    'dynamic': ('deep_gemm', 'vllm.third_party.deep_gemm'),
}
SKIP_NAMESPACES = {'aten', 'prim', 'prims', 'profiler', 'c10d', '_c10d_functional'}

# Native entries that accept empty inputs, each with its evidence. An empty
# input to any other entry fails the test.
EMPTY_SAFE = {
    'vllm.model_executor.kernels.linear.cute_dsl.batch_invariant:bf16_fixed_mm': (
        'returns early for M == 0 (cute_dsl/batch_invariant.py _bf16_fixed_mm); '
        'probe 7997179 bf16_fp32_linear_0rows ok'
    ),
    'vllm::bf16_fixed_mm': 'same op as above, reached through torch.ops.vllm',
    'megatron.lite.primitive.kernels.mxfp8_block32:batched_block32_mm': (
        'zero-row CP state: 26 calls/rank with empty activations, no residual '
        'CUDA error and EP4xCP2 log-probs bitwise equal to CP1 (job 7999396); '
        'probe 7997179 mxfp8_linear_0rows ok'
    ),
    'vllm.model_executor.kernels.batch_invariant:ordered_topk_combine': (
        'zero-row CP state: 4 calls/rank with empty inputs, no residual CUDA '
        'error and EP4xCP2 log-probs bitwise equal to CP1 (job 7999396)'
    ),
}


def _is_native_module(name):
    return name.split('.')[0] in ('vllm', 'deep_gemm') or name.startswith(
        'megatron.lite.primitive.kernels'
    )


class _Scan(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.entries = set()
        # ``deep_gemm`` also arrives as a parameter (w4a8_experts passes the
        # resolved module around), so the bare name always means DeepGEMM.
        self.aliases = {'deep_gemm': 'deep_gemm'}

    def visit_ImportFrom(self, node):
        module = node.module or ''
        if _is_native_module(module):
            for alias in node.names:
                name = alias.asname or alias.name
                # ``from vllm import _custom_ops as ops``: a module, used via ops.f.
                self.aliases[name] = f'{module}.{alias.name}'
                self.entries.add(f'{module}:{alias.name}')
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if _is_native_module(alias.name):
                self.aliases[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_Assign(self, node):
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == 'import_module'
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, str)
            and _is_native_module(value.args[0].value)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = value.args[0].value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == 'import_module'
            and value.args
            and not isinstance(value.args[0], ast.Constant)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = 'dynamic'
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == '_import_deep_gemm'
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = 'deep_gemm'
        self.generic_visit(node)

    def visit_Call(self, node):
        # getattr(alias, 'name', ...) on a native module.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == 'getattr'
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.aliases
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self.entries.add(f'{self.aliases[node.args[0].id]}:{node.args[1].value}')
        self.generic_visit(node)

    def visit_Attribute(self, node):
        chain = []
        current = node
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            chain.append(current.id)
            chain.reverse()
            if chain[:2] == ['torch', 'ops'] and len(chain) >= 4:
                if chain[2] not in SKIP_NAMESPACES:
                    self.entries.add(f'{chain[2]}::{chain[3]}')
            elif chain[0] in self.aliases and len(chain) == 2:
                self.entries.add(f'{self.aliases[chain[0]]}:{chain[1]}')
        self.generic_visit(node)


def scan_native_entries(root=None):
    """Sorted native entry names reachable from MLite source under ``root``."""
    root = Path(root or Path(importlib.import_module('megatron.lite').__file__).parent)
    entries = set()
    for path in sorted(root.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        scan = _Scan(path)
        scan.visit(ast.parse(path.read_text(), str(path)))
        entries |= scan.entries
    return sorted(entries)


def _tensors(value, depth=0):
    if depth > 4:
        return
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item, depth + 1)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item, depth + 1)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _tensors(getattr(value, field.name), depth + 1)


def _resolve(entry):
    """(module object, attribute) for a ``module:attr`` entry, or None."""
    module_name, attr = entry.split(':')
    candidates = DYNAMIC_MODULES.get(module_name, (module_name,))
    for name in candidates:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, attr):
            return module, attr
    return None


@dataclasses.dataclass
class _Record:
    calls: dict = dataclasses.field(default_factory=lambda: defaultdict(int))
    empty: dict = dataclasses.field(default_factory=lambda: defaultdict(int))


class NativeCallTracker(TorchDispatchMode):
    """Wrap scanned entries, watch the dispatcher, attribute calls to CP states."""

    def __init__(self, entries, *, device_type='cuda'):
        super().__init__()
        self.entries = set(entries)
        self.device_type = device_type
        self.state = 'nocp'
        self.depth = 0
        self.records = defaultdict(_Record)
        self.unregistered = defaultdict(int)
        self._restore = []

    # -- attribution -----------------------------------------------------
    def _observe(self, entry, args, kwargs):
        record = self.records[entry]
        record.calls[self.state] += 1
        empty = any(
            t.numel() == 0 and t.device.type == self.device_type
            for t in _tensors((args, kwargs))
        )
        if empty:
            record.empty[self.state] += 1

    def _wrap(self, entry, fn):
        tracker = self

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if tracker.depth == 0:
                tracker._observe(entry, args, kwargs)
            tracker.depth += 1
            try:
                return fn(*args, **kwargs)
            finally:
                tracker.depth -= 1

        wrapper.__b1_native_original__ = fn
        return wrapper

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        namespace = func.namespace
        if self.depth == 0 and namespace not in SKIP_NAMESPACES:
            entry = f'{namespace}::{func._schema.name.split("::")[-1]}'
            if self._registered(entry):
                self._observe(entry, args, kwargs)
            else:
                self.unregistered[(entry, self.state)] += 1
        return func(*args, **kwargs)

    def _registered(self, entry):
        namespace, op = entry.split('::')
        if entry in self.entries or any(
            e.startswith((f'{namespace}:', 'dynamic:')) and e.endswith(f':{op}')
            for e in self.entries
        ):
            return True
        # An op's own autograd backward (e.g. vllm::native_log_softmax_backward)
        # belongs to the registered forward.
        base = op[: -len('_backward')] if op.endswith('_backward') else None
        return base is not None and any(
            e.endswith(f':{base}') or e == f'{namespace}::{base}' for e in self.entries
        )

    # -- install ---------------------------------------------------------
    def install(self):
        originals = {}
        for entry in sorted(self.entries):
            if '::' in entry:
                continue
            resolved = _resolve(entry)
            if resolved is None:
                continue
            module, attr = resolved
            fn = getattr(module, attr)
            if (
                inspect.ismodule(fn)
                or inspect.isclass(fn)
                or not callable(fn)
                or hasattr(fn, '__b1_native_original__')
            ):
                continue
            wrapped = self._wrap(entry, fn)
            originals[id(fn)] = wrapped
            setattr(module, attr, wrapped)
            self._restore.append((module, attr, fn))
        # Module-level ``from m import f`` bindings already made in MLite.
        for name, module in list(sys.modules.items()):
            if not name.startswith('megatron.lite') or module is None:
                continue
            for attr, value in list(vars(module).items()):
                wrapped = originals.get(id(value))
                if wrapped is not None:
                    setattr(module, attr, wrapped)
                    self._restore.append((module, attr, value))
        return self

    def uninstall(self):
        for module, attr, value in reversed(self._restore):
            setattr(module, attr, value)
        self._restore.clear()

    @contextlib.contextmanager
    def document_state(self, state):
        previous, self.state = self.state, state
        try:
            yield
        finally:
            self.state = previous

    # -- verdict ---------------------------------------------------------
    def report(self):
        return dict(
            calls={e: dict(r.calls) for e, r in sorted(self.records.items())},
            empty={
                e: dict(r.empty) for e, r in sorted(self.records.items()) if r.empty
            },
            unregistered={
                f'{e}@{s}': n for (e, s), n in sorted(self.unregistered.items())
            },
        )

    def violations(self):
        out = [
            f'unregistered native call {k} x{v}'
            for k, v in self.report()['unregistered'].items()
        ]
        for entry, states in self.report()['empty'].items():
            if entry not in EMPTY_SAFE:
                out.append(f'empty input to {entry} in states {states}')
        return out


def cp_document_state(document):
    """``zero``/``one``/``part``/``whole``: this rank's rows of one document."""
    local, length = document.local_length, document.length
    if local == 0:
        return 'zero'
    if local == 1:
        return 'one'
    return 'whole' if local == length else 'part'


@contextlib.contextmanager
def track_cp_documents(tracker):
    """Tag tracker calls with the CP state of the document being executed."""
    from megatron.lite.primitive.modules.attention import cp

    original = cp.ContiguousCPSequence.document

    def document(self, begin, end):
        result = original(self, begin, end)
        tracker.state = cp_document_state(result)
        return result

    cp.ContiguousCPSequence.document = document
    try:
        yield tracker
    finally:
        cp.ContiguousCPSequence.document = original
        tracker.state = 'nocp'
