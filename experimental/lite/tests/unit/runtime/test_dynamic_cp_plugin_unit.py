# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.runtime.contracts.data import (
    ForwardResult,
    ModelOutputs,
    PackedBatch,
)
from megatron.lite.runtime.contracts.handle import ModelHandle
from megatron.lite.runtime.contracts.loss import LossContext, split_loss_context


class _Group:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def test_dynamic_cp_exposes_logical_dp_one_without_replacing_physical_dp(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    physical_dp_group = _Group(4, rank=2)
    pool_group = _Group(4, rank=2)
    logical_dp_group = _Group(1)
    ps = SimpleNamespace(
        dp_size=4,
        dp_rank=2,
        dp_group=physical_dp_group,
        dp_cp_group=pool_group,
        dp_cp_size=4,
        cp_size=1,
        pp_size=1,
    )
    handle = ModelHandle(
        model=object(),
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 8},
        create_groups=lambda _ps, _minimum, _parallel: {
            1: logical_dp_group,
            4: pool_group,
        },
    )

    plugin.initialize(handle)

    assert handle.dp_size == 1
    assert handle.dp_rank == 0
    assert handle.dp_group is logical_dp_group
    assert ps.dp_size == 4
    assert ps.dp_rank == 2
    assert ps.dp_group is physical_dp_group


def test_install_wraps_only_the_target_runtime_instance():
    from megatron.lite.runtime.backends.mlite.dynamic_cp import install

    class Runtime:
        def build_model(self):
            return "handle"

        def forward_backward(self, *args, **kwargs):
            return args, kwargs

    enabled = Runtime()
    disabled = Runtime()

    install(enabled, {"max_seqlen_per_dp_cp_rank": 8})

    assert "forward_backward" in enabled.__dict__
    assert "build_model" in enabled.__dict__
    assert "forward_backward" not in disabled.__dict__
    assert "build_model" not in disabled.__dict__


def test_disabled_runtime_does_not_import_dynamic_cp(monkeypatch):
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "megatron.lite.runtime.backends.mlite.dynamic_cp":
            pytest.fail("disabled runtime must not import the Dynamic CP plugin")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    runtime = MegatronLiteRuntime("", MegatronLiteConfig(impl_cfg={"use_thd": True}))

    assert "forward_backward" not in runtime.__dict__
    assert "build_model" not in runtime.__dict__


def test_runtime_config_installs_dynamic_cp_sidecar():
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    runtime = MegatronLiteRuntime(
        "",
        MegatronLiteConfig(
            impl_cfg={
                "use_thd": True,
                "runtime_plugins": {
                    "dynamic_context_parallel": {"max_seqlen_per_dp_cp_rank": 8}
                },
            }
        ),
    )

    assert "forward_backward" in runtime.__dict__
    assert "build_model" in runtime.__dict__


def test_runtime_plugins_must_be_a_mapping():
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    with pytest.raises(TypeError, match="runtime_plugins must be a mapping"):
        MegatronLiteRuntime(
            "", MegatronLiteConfig(impl_cfg={"runtime_plugins": object()})
        )


class _CrossCutScheduler:
    def __init__(self, **kwargs):
        assert kwargs["dp_size"] == 1
        assert kwargs["cp_size"] == 2
        assert kwargs["max_seqlen_per_dp_cp_rank"] == 3

    def get_groups_and_subsamples(self, sample_lengths):
        # Length 5 must cross the CP=2 cut when one rank has capacity 3.
        assert sample_lengths == [(0, 5), (1, 2)]
        return [[[0, 1], [0, 1]]]


class _SplitScheduler:
    def __init__(self, **_kwargs):
        pass

    def get_groups_and_subsamples(self, sample_lengths):
        assert sample_lengths == [(0, 1), (1, 1)]
        return [[[0], [1]]]


def _install_fake_scheduler(monkeypatch):
    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _CrossCutScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)


def test_logical_dp_loss_compensates_physical_pool_average(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _SplitScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    pool, singleton = _Group(2), _Group(1)
    ps = SimpleNamespace(
        dp_size=2,
        dp_rank=0,
        dp_group=pool,
        dp_cp_group=pool,
        cp_size=1,
        cp_rank=0,
        cp_group=singleton,
        pp_size=1,
    )
    handle = ModelHandle(
        model=object(),
        parallel_state=ps,
        config=SimpleNamespace(
            parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
            impl_cfg={"use_thd": True},
        ),
        _extras={"forward_step": lambda *_args: {}},
    )
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 1},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(handle)
    prepared = plugin._prepare(
        handle,
        iter(
            [
                (
                    PackedBatch(
                        input_ids=torch.tensor([1, 2]),
                        labels=torch.tensor([1, 2]),
                        seq_lens=torch.tensor([1, 1]),
                    ),
                    LossContext(source_batch=torch.tensor([[0.0], [0.0]])),
                )
            ]
        ),
        lambda *_args: (torch.tensor(1.0), {}),
        1,
    )
    selected, context = next(prepared.data)
    loss, _ = prepared.loss({}, selected, context)

    assert torch.equal(loss, torch.tensor(2.0))
    prepared.finish(require_complete=True)


def _loss_fn(kind: str, collector: list[dict]):
    def calculate(output, _batch, context):
        values = output["values"]
        source = context.source_batch
        if kind == "sft":
            loss = ((values - source) ** 2).sum()
        else:
            loss = -(values * source).sum()
        if not getattr(calculate, "runtime_collects_outputs", False):
            collector.append(
                {
                    "model_output": {"values": values.detach()},
                    "loss": float(loss.detach()),
                    "metrics": {"kind": kind},
                }
            )
        return loss, {"kind": kind}

    calculate.runtime_output_collector = collector
    calculate.runtime_output_extractor = lambda output: {"values": output["values"]}
    return calculate


def _run_loop(
    handle, data, loss_fn, *, num_microbatches=1, forward_only=False, **_kwargs
):
    loss = None
    for _ in range(num_microbatches):
        batch, context = split_loss_context(next(data))
        output = handle._extras["forward_step"](handle._model, batch)
        loss, _ = loss_fn(output, batch, context)
        if not forward_only:
            loss.backward()
    return ForwardResult(model_output=ModelOutputs(loss=loss.detach()))


@pytest.mark.parametrize(
    "kind,source", [("sft", [[1.0], [3.0]]), ("rl", [[2.0], [-1.0]])]
)
def test_sft_and_rl_true_loss_match_disabled_reference_across_cp_cut(
    monkeypatch, kind, source
):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    _install_fake_scheduler(monkeypatch)
    pool = _Group(2)
    singleton = _Group(1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, records, group: output.__setitem__(slice(None), [records, []]),
    )

    batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        labels=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        seq_lens=torch.tensor([5, 2]),
    )
    source_batch = torch.tensor(source)

    def make_handle(weight):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(weight)
        ps = SimpleNamespace(
            dp_size=2,
            dp_rank=0,
            dp_group=pool,
            dp_cp_group=pool,
            cp_size=1,
            cp_rank=0,
            cp_group=singleton,
            pp_size=1,
        )

        def forward(module, selected):
            offsets = selected.cu_seqlens.tolist()
            rows = [
                selected.input_ids[start:end].float().mean().reshape(1)
                for start, end in zip(offsets[:-1], offsets[1:], strict=True)
            ]
            return {"values": module(torch.stack(rows))}

        return ModelHandle(
            model=model,
            parallel_state=ps,
            config=SimpleNamespace(
                parallel=SimpleNamespace(tp=1, cp=1, pp=1, vpp=1),
                impl_cfg={"use_thd": True},
            ),
            _extras={"forward_step": forward},
        )

    baseline_handle = make_handle(0.25)
    baseline_records = []
    baseline_result = _run_loop(
        baseline_handle,
        iter([(batch, LossContext(source_batch=source_batch))]),
        _loss_fn(kind, baseline_records),
    )

    dcp_handle = make_handle(0.25)
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 3},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(dcp_handle)
    dcp_records = []
    dcp_result = plugin.wrap_forward_backward(_run_loop)(
        dcp_handle,
        iter([(batch, LossContext(source_batch=source_batch))]),
        _loss_fn(kind, dcp_records),
    )

    assert torch.equal(dcp_result.model_output.loss, baseline_result.model_output.loss)
    assert torch.equal(
        dcp_handle._model.weight.grad, baseline_handle._model.weight.grad
    )
    assert torch.equal(
        dcp_records[0]["model_output"]["values"].values(),
        baseline_records[0]["model_output"]["values"].reshape(-1),
    )
