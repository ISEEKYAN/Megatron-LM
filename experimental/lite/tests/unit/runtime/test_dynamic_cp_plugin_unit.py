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


def test_dynamic_cp_split_merge_preserves_jagged_tensordict_samples():
    TensorDict = pytest.importorskip("tensordict").TensorDict
    NonTensorData = pytest.importorskip("tensordict.tensorclass").NonTensorData
    from megatron.lite.runtime.backends.mlite.dynamic_cp import (
        _merge_source,
        _split_source,
    )

    source = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(
                [torch.arange(3), torch.arange(5)], layout=torch.jagged
            ),
            "loss_mask": torch.nested.as_nested_tensor(
                [torch.ones(3), torch.ones(5)], layout=torch.jagged
            ),
            "temperature": torch.tensor([0.5, 0.75]),
            "metadata": ["first", "second"],
            "pad_mode": ["no_padding", "no_padding"],
            "scalar_control": NonTensorData(data=torch.tensor(3), batch_size=[2]),
            "vector_control": NonTensorData(data=torch.tensor([3, 4]), batch_size=[2]),
        },
        batch_size=[2],
    )

    samples = _split_source(source, count=2)
    merged = _merge_source([samples[1], samples[0]], torch.device("cpu"))

    assert merged.batch_size == torch.Size([2])
    assert merged["input_ids"].is_nested
    assert [row.tolist() for row in merged["input_ids"].unbind()] == [
        [0, 1, 2, 3, 4],
        [0, 1, 2],
    ]
    assert [row.tolist() for row in merged["loss_mask"].unbind()] == [
        [1.0] * 5,
        [1.0] * 3,
    ]
    assert torch.equal(merged["temperature"], torch.tensor([0.75, 0.5]))
    assert list(merged["metadata"]) == ["second", "first"]
    assert isinstance(merged.get("pad_mode"), NonTensorData)
    assert merged.get("pad_mode").data == "no_padding"
    assert isinstance(merged.get("scalar_control"), NonTensorData)
    assert torch.equal(merged.get("scalar_control").data, torch.tensor(3))
    assert isinstance(merged.get("vector_control"), NonTensorData)
    assert torch.equal(merged.get("vector_control").data, torch.tensor([3, 4]))


def test_dynamic_cp_exposes_logical_dp_one_without_replacing_physical_dp(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    physical_dp_group = _Group(4, rank=2)
    pool_group = _Group(4, rank=2)
    logical_dp_group = _Group(1)
    cp2_group = _Group(2)
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
            2: cp2_group,
            4: pool_group,
        },
    )

    plugin.initialize(handle)

    assert handle.dp_size == 1
    assert handle.dp_rank == 0
    assert handle.dp_group is logical_dp_group
    assert handle.metric_group is pool_group
    assert ps.dp_size == 4
    assert ps.dp_rank == 2
    assert ps.dp_group is physical_dp_group


def test_dynamic_cp_eagerly_initializes_each_non_singleton_group():
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    pool, singleton, cp2 = _Group(4), _Group(1), _Group(2)
    ps = SimpleNamespace(
        dp_size=4,
        dp_rank=0,
        dp_group=pool,
        dp_cp_group=pool,
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
    initialized = []
    plugin = DynamicCPPlugin(
        {"max_seqlen_per_dp_cp_rank": 8},
        create_groups=lambda _ps, _minimum, _parallel: {
            1: singleton,
            2: cp2,
            4: pool,
        },
        initialize_group=initialized.append,
    )

    plugin.initialize(handle)

    assert initialized == [cp2, pool]


def test_dynamic_cp_group_initializer_uses_batched_ring_as_first_p2p(monkeypatch):
    from megatron.lite.runtime.backends.mlite import dynamic_cp

    group = _Group(4)
    token = object()
    events = []

    class _Work:
        def wait(self):
            events.append("wait")

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda actual: [4, 6, 8, 10]
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 6)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: token)
    monkeypatch.setattr(
        torch.distributed,
        "P2POp",
        lambda op, tensor, peer, actual: (op, tensor, peer, actual),
    )
    monkeypatch.setattr(
        torch.distributed,
        "batch_isend_irecv",
        lambda ops: events.append(("batch", ops)) or [_Work(), _Work()],
    )
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda **kwargs: events.append(("barrier", kwargs)),
    )

    dynamic_cp._initialize_group(group)

    assert events == [
        (
            "batch",
            [
                (torch.distributed.irecv, token, 4, group),
                (torch.distributed.isend, token, 8, group),
            ],
        ),
        "wait",
        "wait",
        ("barrier", {"group": group, "device_ids": [2]}),
    ]


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


class _MixedCPScheduler:
    def __init__(self, **_kwargs):
        pass

    def get_groups_and_subsamples(self, sample_lengths):
        assert sample_lengths == [(0, 2), (1, 8), (2, 2)]
        return [[[0], [2]], [[1], [1]]]


class _CP4OnlyScheduler:
    def __init__(self, **_kwargs):
        pass

    def get_groups_and_subsamples(self, sample_lengths):
        assert sample_lengths == [(0, 16)]
        return [[[0], [0], [0], [0]]]


def _install_fake_scheduler(monkeypatch):
    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _CrossCutScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)


def test_mixed_cp_plan_reports_each_sample_group_and_histogram(monkeypatch, capsys):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _MixedCPScheduler
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
        {"max_seqlen_per_dp_cp_rank": 4},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
    )
    plugin.initialize(handle)
    prepared = plugin._prepare(
        handle,
        PackedBatch(
            input_ids=torch.arange(12),
            labels=torch.arange(12),
            seq_lens=torch.tensor([2, 8, 2]),
        ),
        None,
        1,
    )

    local_cp_sizes = []
    for item in prepared.data:
        selected, _context = split_loss_context(item)
        local_cp_sizes.append(selected.extras["_mlite_dcp_local_cp_size"])
    prepared.finish(require_complete=True)

    assert local_cp_sizes == [1, 2]
    assert capsys.readouterr().out.splitlines() == [
        "MLITE_DYNAMIC_CP_PLAN step=0 cp_size_space=[1,2] "
        'cp_size_histogram={"1":2,"2":1} '
        'groups=[{"cp_size":1,"ranks":[0],"sample_ids":[0]},'
        '{"cp_size":1,"ranks":[1],"sample_ids":[2]},'
        '{"cp_size":2,"ranks":[0,1],"sample_ids":[1]}] global_num_tokens=None'
    ]


def test_mixed_cp_uses_pool_global_token_count_for_loss_normalization(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _MixedCPScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)

    def run_rank(rank):
        pool, singleton = _Group(2, rank), _Group(1)

        def sum_uneven_owned_tokens(value, *, group):
            assert group is pool
            owned = 7 if rank == 0 else 1
            peer = 1 if rank == 0 else 7
            assert torch.equal(value, torch.tensor(owned, dtype=torch.int64))
            value.add_(peer)

        monkeypatch.setattr(torch.distributed, "all_reduce", sum_uneven_owned_tokens)
        ps = SimpleNamespace(
            dp_size=2,
            dp_rank=rank,
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
            {"max_seqlen_per_dp_cp_rank": 4},
            create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: pool},
        )
        plugin.initialize(handle)
        prepared = plugin._prepare(
            handle,
            iter(
                [
                    (
                        PackedBatch(
                            input_ids=torch.arange(12),
                            labels=torch.arange(12),
                            seq_lens=torch.tensor([2, 8, 2]),
                            loss_mask=torch.tensor(
                                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
                            ),
                        ),
                        LossContext(loss_scale=1 / 4, source_batch=torch.zeros(3, 1)),
                    )
                ]
            ),
            lambda _output, selected, context: (
                selected.loss_mask.sum(dtype=torch.float32) * context.loss_scale,
                {},
            ),
            1,
        )
        losses = []
        token_counts = []
        for item in prepared.data:
            selected, context = split_loss_context(item)
            loss, _metrics = prepared.loss({}, selected, context)
            losses.append(loss)
            token_counts.append(selected.extras["_mlite_dcp_global_num_tokens"])
        prepared.finish(require_complete=True)
        return sum(losses) / len(losses), token_counts

    rank0_loss, rank0_tokens = run_rank(0)
    rank1_loss, rank1_tokens = run_rank(1)
    dcp_global_loss = (rank0_loss + rank1_loss) / 2
    baseline_tokens = torch.tensor(7, dtype=torch.int64) + torch.tensor(
        1, dtype=torch.int64
    )
    baseline_global_loss = torch.tensor(8.0) / baseline_tokens

    assert rank0_tokens == rank1_tokens == [8, 8]
    assert torch.equal(baseline_tokens, torch.tensor(8, dtype=torch.int64))
    assert torch.equal(dcp_global_loss, baseline_global_loss)


@pytest.mark.parametrize("with_context", [False, True])
def test_dynamic_cp_missing_loss_mask_fails_before_normalization_can_degrade(
    monkeypatch, with_context
):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _SplitScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    pool, singleton = _Group(2), _Group(1)
    handle = ModelHandle(
        model=object(),
        parallel_state=SimpleNamespace(
            dp_size=2,
            dp_rank=0,
            dp_group=pool,
            dp_cp_group=pool,
            cp_size=1,
            cp_rank=0,
            cp_group=singleton,
            pp_size=1,
        ),
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

    with pytest.raises(
        ValueError,
        match="requires loss_mask on every sample for pool-global loss normalization",
    ):
        plugin._prepare(
            handle,
            iter(
                [
                    (
                        PackedBatch(
                            input_ids=torch.tensor([1, 2]),
                            labels=torch.tensor([1, 2]),
                            seq_lens=torch.tensor([1, 1]),
                        ),
                        (
                            LossContext(
                                loss_scale=0.5,
                                source_batch=torch.tensor([[0.0], [0.0]]),
                            )
                            if with_context
                            else None
                        ),
                    )
                ]
            ),
            lambda *_args: (torch.tensor(1.0), {}),
            1,
        )


def test_required_cp_size_coverage_fails_loudly(monkeypatch):
    from megatron.lite.runtime.backends.mlite.dynamic_cp import DynamicCPPlugin

    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.DefaultDynamicCPScheduler = _CP4OnlyScheduler
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)
    pool, singleton, cp2 = _Group(4), _Group(1), _Group(2)
    ps = SimpleNamespace(
        dp_size=4,
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
        {"max_seqlen_per_dp_cp_rank": 4, "require_full_cp_size_coverage": True},
        create_groups=lambda _ps, _minimum, _parallel: {1: singleton, 2: cp2, 4: pool},
    )
    plugin.initialize(handle)

    with pytest.raises(
        RuntimeError,
        match=r"did not cover required cp_size values \[1, 2\]; expected \[1, 2, 4\]",
    ):
        plugin._prepare(
            handle,
            PackedBatch(
                input_ids=torch.arange(16),
                labels=torch.arange(16),
                seq_lens=torch.tensor([16]),
            ),
            None,
            1,
        )


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

    def sum_pool_tokens(value, *, group):
        assert group is pool
        value.add_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", sum_pool_tokens)
    prepared = plugin._prepare(
        handle,
        iter(
            [
                (
                    PackedBatch(
                        input_ids=torch.tensor([1, 2]),
                        labels=torch.tensor([1, 2]),
                        seq_lens=torch.tensor([1, 1]),
                        loss_mask=torch.ones(2),
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

    # Two one-token leaders contribute one normalized global loss after the
    # physical-pool DDP average; neither rank silently keeps a half-loss.
    assert torch.equal(loss, torch.tensor(1.0))
    prepared.finish(require_complete=True)


def test_restore_outputs_merges_metrics_from_every_distinct_leader(monkeypatch):
    from megatron.lite.runtime.backends.mlite import dynamic_cp

    pool = _Group(2)
    rank0_records = [
        {
            "sample_ids": [0],
            "model_output": {"values": [torch.tensor([10.0])]},
            "loss": 1.0,
            "metrics": {"score": 1.0},
        }
    ]
    rank1_records = [
        {
            "sample_ids": [1],
            "model_output": {"values": [torch.tensor([30.0])]},
            "loss": 3.0,
            "metrics": {"score": 3.0},
        }
    ]

    def gather(output, records, *, group):
        assert group is pool
        assert records is rank0_records
        output[:] = [rank0_records, rank1_records]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    collector = []
    dynamic_cp._restore_outputs(
        collector,
        rank0_records,
        pool=pool,
        input_groups=[[0, 1]],
        device=torch.device("cpu"),
    )

    assert len(collector) == 1
    scores = collector[0]["metrics"]["score"]
    assert scores == [1.0, 3.0]
    assert sum(scores) / len(scores) == 2.0
    assert sum(scores) / len(scores) not in scores
    assert sum(item.get("loss", 0.0) for item in collector) == 4.0


def test_restore_outputs_preserves_metric_aggregator_across_leaders(monkeypatch):
    from megatron.lite.runtime.backends.mlite import dynamic_cp

    class Metric:
        def __init__(self, values):
            self.values = list(values)

        def init_list(self):
            return Metric([])

        def append(self, value):
            self.values.extend(value.values)

    pool = _Group(2)
    rank0_records = [
        {
            "sample_ids": [0],
            "model_output": {},
            "loss": 0.0,
            "metrics": {"score": Metric([1.0])},
        }
    ]
    rank1_records = [
        {
            "sample_ids": [1],
            "model_output": {},
            "loss": 0.0,
            "metrics": {"score": Metric([3.0])},
        }
    ]
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, _records, group: output.__setitem__(
            slice(None), [rank0_records, rank1_records]
        ),
    )
    collector = []

    dynamic_cp._restore_outputs(
        collector,
        rank0_records,
        pool=pool,
        input_groups=[[0, 1]],
        device=torch.device("cpu"),
    )

    assert collector[0]["metrics"]["score"].values == [1.0, 3.0]


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
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, group: None)

    batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        labels=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        seq_lens=torch.tensor([5, 2]),
        loss_mask=torch.ones(7),
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
        iter([(batch, LossContext(loss_scale=1 / 7, source_batch=source_batch))]),
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
        iter([(batch, LossContext(loss_scale=1 / 7, source_batch=source_batch))]),
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
