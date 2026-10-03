# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint end-to-end on real Qwen assembly, with CPU TE storage providers.

TE compute kernels are not emulated; these tests exercise inventory, codecs,
QAT master ownership and checkpoint I/O, not forward/backward parity.
"""

import shutil

import pytest
import torch
from megatron.lite.primitive.ckpt.hf_weights import (
    BoundedTensorReader,
    stream_export_to_shards,
)
from megatron.lite.primitive.quantization.qat import QATSpec, apply_qat_to_chunks
from test_all_model_qat_r3_contracts import _install_cpu_te_construction_stubs


def make_qwen(monkeypatch, transformer_engine_import_stub, vocab_size=1024):
    _install_cpu_te_construction_stubs(transformer_engine_import_stub, monkeypatch)
    import transformer_engine.pytorch as cpu_te
    from megatron.lite.primitive import transformer_engine as te

    monkeypatch.setattr(te, "_TE", cpu_te)
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel
    from megatron.lite.primitive.parallel import ParallelState

    config = Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=vocab_size,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        layer_types=["full_attention"],
    )
    model = Qwen3MoEModel(config, ParallelState()).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.8, 0.8)
    return model


@pytest.mark.parametrize("target", ["hf", "mxfp4"])
def test_qwen_load_save_export_and_exact_qat_masters(
    tmp_path, monkeypatch, transformer_engine_import_stub, target
):
    from megatron.lite.model.qwen3_moe.lite.checkpoint import Qwen3MoEBoundSpec

    torch.manual_seed(12)
    model = make_qwen(monkeypatch, transformer_engine_import_stub)
    from megatron.lite.model.qwen3_moe.lite import protocol

    apply_qat_to_chunks([model], QATSpec(enabled=True, format="mxfp4", group_size=32))
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    bindings = model.checkpoint_bindings()
    assert {id(b.tensor) for b in bindings} == {id(p) for p in model.parameters()}
    assert any("parametrizations" in name for name in original)
    budget = 4 * 1024**2
    protocol.save_hf_weights(
        [model],
        str(tmp_path),
        model.config,
        model.ps,
        bounded=True,
        target=target,
        buffer_max_size_bytes=budget,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    protocol.load_hf_weights(
        model,
        str(tmp_path),
        model.config,
        model.ps,
        bounded=True,
        buffer_max_size_bytes=budget,
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name]), name

    # External HF loading must also work without the exact-training sidecar.
    shutil.rmtree(tmp_path / "mlite_masters")
    reader = BoundedTensorReader(str(tmp_path))
    spec = Qwen3MoEBoundSpec(model.config, model.ps, target)
    expected = {
        b.name: spec.hf_to_native(
            b.name, [spec.decode(reader, k, budget) for k in b.sources]
        )
        for b in bindings
    }
    protocol.load_hf_weights(
        model,
        str(tmp_path),
        model.config,
        model.ps,
        bounded=True,
        buffer_max_size_bytes=budget,
    )
    for binding in model.checkpoint_bindings():
        assert torch.equal(binding.tensor, expected[binding.name]), binding.name
    second = tmp_path / "second"
    stream_export_to_shards(
        protocol.export_hf_weights(
            [model],
            model.config,
            model.ps,
            bounded=True,
            target=target,
            buffer_max_size_bytes=budget,
        ),
        str(second),
    )
    reloaded = BoundedTensorReader(str(second))
    assert reloaded.keys() == reader.keys()
    # MXFP4 can choose a different equivalent exponent after dequantization.
    # Numerical release values must survive; archival byte passthrough is tested
    # separately, while the training sidecar above is bitwise exact.
    for binding in bindings:
        for name in binding.sources:
            assert torch.equal(
                spec.decode(reader, name, budget), spec.decode(reloaded, name, budget)
            ), name


def test_qwen_rejects_unsupported_parallel_and_unbound_owner(
    tmp_path, monkeypatch, transformer_engine_import_stub
):
    from megatron.lite.model.qwen3_moe.lite.checkpoint import save_hf_weights

    model = make_qwen(monkeypatch, transformer_engine_import_stub)
    model.ps.tp_size = 2
    with pytest.raises(NotImplementedError, match="TP=EP"):
        save_hf_weights(
            model,
            str(tmp_path),
            model.config,
            model.ps,
            bounded=True,
            buffer_max_size_bytes=1024**2,
        )
    model.ps.tp_size = 1
    model.register_parameter("unexpected", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="exactly one"):
        model.checkpoint_bindings()


def test_row_load_never_requests_complete_embedding(
    tmp_path, monkeypatch, transformer_engine_import_stub
):
    from megatron.lite.model.qwen3_moe.lite.checkpoint import (
        load_hf_weights,
        save_hf_weights,
    )

    model = make_qwen(monkeypatch, transformer_engine_import_stub, vocab_size=32768)
    budget = 4 * 1024**2
    save_hf_weights(
        model,
        str(tmp_path),
        model.config,
        model.ps,
        bounded=True,
        buffer_max_size_bytes=budget,
    )
    read = BoundedTensorReader.read

    def bounded_read(self, name, budget, rows=None):
        if "embed" in name or "head" in name:
            assert rows is not None
            assert rows[1] - rows[0] < model.config.vocab_size
        return read(self, name, budget, rows)

    monkeypatch.setattr(BoundedTensorReader, "read", bounded_read)
    load_hf_weights(
        model,
        str(tmp_path),
        model.config,
        model.ps,
        bounded=True,
        buffer_max_size_bytes=budget,
    )


def _rss_worker(rank, counts, root):
    import gc
    import importlib.util
    import json
    import resource
    import threading
    from pathlib import Path

    from megatron.lite.model.qwen3_moe.lite.checkpoint import (
        load_hf_weights,
        save_hf_weights,
    )

    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        'bounded_conftest', Path(__file__).parents[2] / 'conftest.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patch = pytest.MonkeyPatch()
    install = module.transformer_engine_import_stub.__wrapped__(patch)
    budget = 4 * 1024**2
    path = Path(root) / str(rank)
    warm = make_qwen(patch, install)
    save_hf_weights(
        warm,
        str(path),
        warm.config,
        warm.ps,
        bounded=True,
        buffer_max_size_bytes=budget,
    )
    load_hf_weights(
        warm,
        str(path),
        warm.config,
        warm.ps,
        bounded=True,
        buffer_max_size_bytes=budget,
    )
    del warm
    gc.collect()
    model = make_qwen(patch, install, counts[rank])

    def rss():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
        raise AssertionError('Linux RSS unavailable')

    baseline = rss()
    samples = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.002):
            samples[0] = max(samples[0], rss())

    thread = threading.Thread(target=sample)
    thread.start()
    try:
        save_hf_weights(
            model,
            str(path),
            model.config,
            model.ps,
            bounded=True,
            buffer_max_size_bytes=budget,
        )
        load_hf_weights(
            model,
            str(path),
            model.config,
            model.ps,
            bounded=True,
            buffer_max_size_bytes=budget,
        )
    finally:
        stop.set()
        thread.join()
    result = dict(
        rows=counts[rank],
        budget=budget,
        processes=1,
        baseline=baseline,
        peak=max(samples[0], rss()),
        maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    result['increment'] = result['peak'] - baseline
    (Path(root) / f'rss-{rank}.json').write_text(json.dumps(result))


def test_qwen_independent_process_host_rss_is_bounded(tmp_path):
    import json

    import torch.multiprocessing as mp

    counts = (16384, 131072, 524288)
    mp.spawn(_rss_worker, args=(counts, str(tmp_path)), nprocs=3, join=True)
    readings = [
        json.loads((tmp_path / f'rss-{rank}.json').read_text()) for rank in range(3)
    ]
    print('QWEN_BOUNDED_RSS ' + json.dumps(readings))
    # Resident embedding/head bytes grow by 248 MiB. Check additional host RSS,
    # allowing fixed allocator/file-I/O overhead; this is not ATen storage.
    assert max(r['increment'] for r in readings) < 24 * 1024**2
    assert readings[-1]['increment'] - readings[0]['increment'] < 8 * 1024**2
