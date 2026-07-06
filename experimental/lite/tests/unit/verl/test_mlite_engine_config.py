# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import MegatronLiteEngine, _build_lr_scheduler
from megatron.lite.runtime.contracts import LossContext


def _optimizer_config(**override_optimizer_config) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1e-6,
        min_lr=None,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.1,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=10,
        lr_warmup_steps=0,
        lr_warmup_init=0.0,
        lr_decay_steps=None,
        lr_decay_style="constant",
        weight_decay_incr_style="constant",
        lr_wsd_decay_style="exponential",
        lr_wsd_decay_steps=None,
        use_checkpoint_opt_param_scheduler=False,
        betas=(0.9, 0.95),
        override_optimizer_config=override_optimizer_config,
    )


def _engine(
    *, engine_config: MegatronLiteEngineConfig, optimizer_config: SimpleNamespace | None = None
) -> MegatronLiteEngine:
    return MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=engine_config,
        optimizer_config=optimizer_config or _optimizer_config(),
        checkpoint_config={},
    )


def _engine_config(**kwargs) -> MegatronLiteEngineConfig:
    values = {"custom_backend_module": None, "impl_cfg": {"use_thd": True}}
    values.update(kwargs)
    return MegatronLiteEngineConfig(**values)


@pytest.mark.parametrize("num_microbatches", [1, 4])
def test_verl_loss_hook_preserves_gradient_and_micro_outputs(num_microbatches):
    engine = _engine(engine_config=_engine_config())
    weight = torch.nn.Parameter(torch.tensor(1.0))
    outputs = []
    engine._build_verl_model_output = lambda **_kwargs: {"log_probs": weight * 3}
    engine.get_data_parallel_group = lambda: None

    hook = engine._make_runtime_loss_fn(
        lambda model_output, **_kwargs: (model_output["log_probs"] / num_microbatches, {}),
        num_microbatches=num_microbatches,
        output_lst=outputs,
    )
    for _ in range(num_microbatches):
        loss, _ = hook({}, object(), LossContext(source_batch=object()))
        (loss / num_microbatches).backward()

    torch.testing.assert_close(weight.grad, torch.tensor(3.0))
    assert [output["loss"] for output in outputs] == [3.0 / num_microbatches] * num_microbatches


def test_optimizer_offload_enables_full_optimizer_state_offload_by_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(
            use_precision_aware_optimizer=True, decoupled_weight_decay=True
        ),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0
    assert optimizer.use_precision_aware_optimizer is True
    assert optimizer.decoupled_weight_decay is True
    assert optimizer.adam_beta1 == 0.9
    assert optimizer.adam_beta2 == 0.95


def test_explicit_optimizer_offload_fraction_overrides_engine_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(offload_fraction=0.25),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 0.25


def test_optimizer_cpu_offload_alias_maps_to_full_offload_fraction() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=False),
        optimizer_config=_optimizer_config(optimizer_cpu_offload=True),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0


def test_mlite_config_threads_rl_parallel_and_impl_settings() -> None:
    engine = _engine(
        engine_config=_engine_config(
            tp=2,
            ep=8,
            etp=1,
            pp=1,
            cp=1,
            optimizer_offload=True,
            attention_backend_override="flash",
            impl_cfg={"use_thd": True, "deterministic": False},
        )
    )

    config = engine._build_mlite_config()

    assert config.model_name == "qwen3_5"
    assert config.impl == "lite"
    assert config.parallel.tp == 2
    assert config.parallel.ep == 8
    assert config.parallel.etp == 1
    assert config.optimizer.offload_fraction == 1.0
    assert config.attention_backend_override == "flash"
    assert config.impl_cfg["use_thd"] is True
    assert config.impl_cfg["deterministic"] is False


def test_local_lr_scheduler_warmup_decay_and_state_roundtrip() -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.1}])
    opt = SimpleNamespace(
        total_training_steps=4,
        lr_warmup_steps=1,
        lr_warmup_steps_ratio=0.0,
        lr_warmup_init=0.0,
        lr=1.0,
        min_lr=0.1,
        lr_decay_steps=4,
        lr_decay_style="linear",
        weight_decay=0.1,
        weight_decay_incr_style="constant",
        lr_wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
    )

    scheduler = _build_lr_scheduler(optimizer, opt)

    assert optimizer.param_groups[0]["lr"] == 0.0
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == 1.0
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.7)

    state = scheduler.state_dict()
    scheduler.step(10)
    scheduler.load_state_dict(state)

    assert scheduler.state_dict() == state
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.7)

def test_engine_export_merges_lora_when_adapter_enabled(monkeypatch) -> None:
    from types import SimpleNamespace

    import verl_mlite.engine.mlite_engine as engine_mod
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    # CUDA cache hygiene is irrelevant to the export-kwargs contract under test
    monkeypatch.setattr(engine_mod, "aggressive_empty_cache", lambda **_: None)

    captured: dict = {}

    class _FakeRuntime:
        @staticmethod
        def export_weights(handle, **kwargs):
            captured.update(kwargs)
            return iter(())

    def make(lora):
        engine = MegatronLiteEngine.__new__(MegatronLiteEngine)
        engine._require_initialized = lambda: None
        # is_param_offload_enabled is a property over engine_config offload flags
        engine.engine_config = SimpleNamespace(
            model_name="qwen2", export_dtype=None, param_offload=False
        )
        engine._mlite_config = SimpleNamespace(impl_cfg={"lora": lora} if lora else {})
        engine.runtime = _FakeRuntime()
        engine.handle = object()
        return engine

    captured.clear()
    make({"rank": 16, "alpha": 32}).get_per_tensor_param()
    assert captured.get("merge_lora") is True

    captured.clear()
    make(None).get_per_tensor_param()
    assert "merge_lora" not in captured

    captured.clear()
    make({"rank": 0}).get_per_tensor_param()
    assert "merge_lora" not in captured


def test_router_replay_config_flag_defaults_off() -> None:
    from verl_mlite.engine.config import MegatronLiteEngineConfig

    assert MegatronLiteEngineConfig().router_replay is False
    assert MegatronLiteEngineConfig(router_replay=True).router_replay is True


def test_router_replay_source_config_gate() -> None:
    # WS2 (R3 phase-2 plan §2.2): "rollout" switches replay to the serving engine's
    # batch-carried routing; default stays the phase-1 self-record proxy. The gate is
    # validated at config time — rollout routing without attached routers is useless.
    from verl_mlite.engine.config import MegatronLiteEngineConfig

    assert MegatronLiteEngineConfig().router_replay_source == "self_record"
    cfg = MegatronLiteEngineConfig(router_replay=True, router_replay_source="rollout")
    assert cfg.router_replay_source == "rollout"
    with pytest.raises(ValueError, match="requires router_replay=True"):
        MegatronLiteEngineConfig(router_replay_source="rollout")
    with pytest.raises(ValueError, match="router_replay_source"):
        MegatronLiteEngineConfig(router_replay=True, router_replay_source="serving")


def _stub_router_replay_module(monkeypatch, calls: list[tuple]):
    import sys
    import types

    class _FakeReplay:
        recorded = [{0: "l0"}, {0: "l1"}]

        @staticmethod
        def clear_global_indices():
            calls.append(("clear",))

        @staticmethod
        def set_global_router_replay_action(action):
            calls.append(("action", getattr(action, "name", None)))

        @staticmethod
        def get_recorded_data():
            return list(_FakeReplay.recorded)

        @staticmethod
        def set_replay_data(data):
            calls.append(("data", len(data)))

        @staticmethod
        def load_microbatch_schedule(schedule):
            calls.append(("schedule", list(schedule)))

        @staticmethod
        def clear_microbatch_schedule():
            calls.append(("clear_schedule",))

        @staticmethod
        def assert_backward_replay_drained():
            calls.append(("drained",))

        @staticmethod
        def set_replay_data_for_microbatch(micro_idx, data):
            calls.append(("mb_data", micro_idx, len(data)))

    class _FakeAction:
        class RECORD:
            name = "RECORD"

        class REPLAY_FORWARD:
            name = "REPLAY_FORWARD"

    stub = types.ModuleType("megatron.lite.primitive.modules.router")
    stub.RouterReplay = _FakeReplay
    stub.RouterReplayAction = _FakeAction
    monkeypatch.setitem(sys.modules, "megatron.lite.primitive.modules.router", stub)
    return _FakeReplay


def _replay_capable_engine(calls: list[tuple]):
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    engine = MegatronLiteEngine.__new__(MegatronLiteEngine)
    engine._router_replay_count = 2
    engine._router_replay_mode = None
    engine._router_replay_routing = None
    engine._router_replay_layout = None
    engine._require_initialized = lambda: None
    engine.engine_config = SimpleNamespace(cp=1)
    engine.forward_backward_batch = lambda data, loss_fn, forward_only=False: calls.append(
        ("fwd", forward_only)
    )
    return engine


def test_engine_record_and_replay_drive_router_replay_actions(monkeypatch) -> None:
    # Engine-side contract only (the runtime-level closed loop is covered by
    # tests/smoke/primitive/test_router_replay_smoke.py): record arms RECORD around a
    # forward-only pass and folds the registry data per sample; replay arms
    # REPLAY_FORWARD, asserts the backward FIFO drained on clean exit, and always
    # clears indices + schedule, even on error.
    calls: list[tuple] = []
    _stub_router_replay_module(monkeypatch, calls)
    engine = _replay_capable_engine(calls)
    engine._fold_recorded_routing = lambda recorded: recorded

    recorded = engine.record_routed_experts({"x": 1})
    assert recorded == [{0: "l0"}, {0: "l1"}]
    assert calls == [
        ("clear",),
        ("action", "RECORD"),
        ("fwd", True),
        ("action", None),
        ("clear",),
        ("clear_schedule",),
    ]
    assert engine._router_replay_mode is None

    # phase-1 compat: per-layer [tokens, K] tensors pin microbatch 0 via set_replay_data
    calls.clear()
    with engine.replay_routed_experts([torch.zeros(4, 2, dtype=torch.long)] * 2):
        calls.append(("inside", engine._router_replay_mode))
    assert calls == [
        ("data", 2),
        ("action", "REPLAY_FORWARD"),
        ("inside", "replay"),
        ("drained",),
        ("action", None),
        ("clear",),
        ("clear_schedule",),
    ]
    assert engine._router_replay_mode is None

    engine._router_replay_count = 0
    try:
        engine.record_routed_experts({"x": 1})
    except RuntimeError as e:
        assert "router_replay" in str(e)
    else:
        raise AssertionError("expected RuntimeError without attached replay")


def test_engine_replay_accepts_per_sample_routing_and_requires_cp1(monkeypatch) -> None:
    # Per-sample [T_i, L, K] routing (record_routed_experts output) is stored to ride
    # the batch instead of being fanned out globally; cp>1 is rejected loudly because
    # routing tensors are not CP-split yet (R3 phase-2 plan §2.4).
    calls: list[tuple] = []
    _stub_router_replay_module(monkeypatch, calls)
    engine = _replay_capable_engine(calls)

    per_sample = [torch.zeros(t, 2, 8, dtype=torch.long) for t in (3, 5)]
    with engine.replay_routed_experts(per_sample):
        routing = engine._router_replay_routing
        assert routing is not None and routing.is_nested
        assert engine._router_replay_mode == "replay"
    assert engine._router_replay_routing is None
    assert ("data", 2) not in calls  # per-sample path must not preload global targets
    assert calls[-3:] == [("action", None), ("clear",), ("clear_schedule",)]

    engine.engine_config = SimpleNamespace(cp=2)
    with pytest.raises(NotImplementedError, match="cp=1"):
        with engine.replay_routed_experts(per_sample):
            pass


def test_forward_backward_auto_arms_replay_only_for_rollout_source() -> None:
    # Rollout-route ingest: with router_replay_source="rollout" every forward_backward_batch over a
    # rollout batch replays the batch-carried routing; a rollout-sourced engine that
    # gets a batch WITHOUT routing fails loudly (training un-replayed would be silently
    # wrong); self_record and already-armed passes never auto-arm.
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    engine = MegatronLiteEngine.__new__(MegatronLiteEngine)
    engine._router_replay_mode = None
    engine.engine_config = SimpleNamespace(router_replay_source="rollout")
    assert engine._should_replay_rollout_routing({"routed_experts": object()}) is True
    with pytest.raises(ValueError, match="routed_experts"):
        engine._should_replay_rollout_routing({"input_ids": object()})

    engine._router_replay_mode = "replay"  # re-entered pass: don't arm twice
    assert engine._should_replay_rollout_routing({"routed_experts": object()}) is False
    engine._router_replay_mode = "record"  # self-record proxy pass over a rollout batch
    assert engine._should_replay_rollout_routing({"routed_experts": object()}) is False

    engine._router_replay_mode = None
    engine.engine_config = SimpleNamespace(router_replay_source="self_record")
    assert engine._should_replay_rollout_routing({"routed_experts": object()}) is False


def _nested(rows):
    return torch.nested.as_nested_tensor(list(rows), layout=torch.jagged)


def _rollout_ingest_engine(*, routers=2, topk=2, num_experts=8):
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    engine = MegatronLiteEngine.__new__(MegatronLiteEngine)
    engine._require_initialized = lambda: None
    engine.engine_config = SimpleNamespace(cp=1)
    engine._router_replay_count = routers
    engine._router_replay_topk = topk
    engine._router_replay_num_experts = num_experts
    engine._router_replay_unmappable_frac = None
    return engine


def test_engine_ingest_masks_unmappable_rollout_routing() -> None:
    # Unmappable-routing masking contract (arXiv:2605.13779 §6.3): the agent loop ZERO-fills routing outside
    # the captured span and 0 is a valid expert id — ingest converts all-zero [L, K]
    # rows to sentinel -1 (live-routing fallback in the router), counts the fraction,
    # and keeps rows that merely contain expert 0. Loss-carrying unmappable tokens
    # trigger the one-shot coverage warning.
    engine = _rollout_ingest_engine()
    input_ids = _nested([torch.arange(3), torch.arange(4)])
    r0 = torch.tensor(
        [[[1, 2], [3, 4]], [[0, 1], [2, 3]], [[0, 0], [0, 0]]], dtype=torch.uint8
    )
    r1 = torch.tensor(
        [[[5, 6], [7, 1]], [[0, 0], [0, 0]], [[2, 3], [4, 5]], [[1, 0], [3, 2]]],
        dtype=torch.uint8,
    )
    loss_mask = _nested([torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 1.0, 1.0])])
    data = {"routed_experts": _nested([r0, r1]), "input_ids": input_ids, "loss_mask": loss_mask}

    sanitized = engine._ingest_rollout_routing(data)

    assert sanitized.is_nested and sanitized.values().dtype == torch.int16
    assert torch.equal(sanitized.offsets().long(), input_ids.offsets().long())
    values = sanitized.values()
    assert (values[2] == -1).all() and (values[4] == -1).all()  # zero-filled rows masked
    assert torch.equal(values[1], r0[1].to(torch.int16))  # expert 0 among others survives
    assert engine._router_replay_unmappable_frac == pytest.approx(2 / 7)
    # sample 0 token 2 is unmappable AND loss-carrying -> one-shot warning latched
    assert engine._warned_unmappable_loss is True


def test_engine_ingest_rejects_mismatched_rollout_routing() -> None:
    # §2.2.4 fail-loudly validation: router count (PP-sliced layouts unsupported),
    # top-k width, expert-id range, non-jagged layout, and token-span misalignment.
    input_ids = _nested([torch.arange(3)])
    good = _nested([torch.tensor([[[1, 2], [3, 4]]] * 3, dtype=torch.uint8)])

    with pytest.raises(ValueError, match=r"L=2.*does not match"):
        _rollout_ingest_engine(routers=3)._ingest_rollout_routing(
            {"routed_experts": good, "input_ids": input_ids}
        )
    with pytest.raises(ValueError, match=r"K=2.*does not match"):
        _rollout_ingest_engine(topk=4)._ingest_rollout_routing(
            {"routed_experts": good, "input_ids": input_ids}
        )
    with pytest.raises(ValueError, match="num_experts=4"):
        _rollout_ingest_engine(num_experts=4)._ingest_rollout_routing(
            {"routed_experts": good, "input_ids": input_ids}
        )
    with pytest.raises(ValueError, match="jagged"):
        _rollout_ingest_engine()._ingest_rollout_routing(
            {"routed_experts": torch.zeros(1, 3, 2, 2), "input_ids": input_ids}
        )
    with pytest.raises(ValueError, match="token spans"):
        _rollout_ingest_engine()._ingest_rollout_routing(
            {
                "routed_experts": _nested([torch.ones(2, 2, 2, dtype=torch.uint8)]),
                "input_ids": input_ids,
            }
        )


def test_loss_hook_reports_unmappable_frac_metric() -> None:
    # The the unmappable-routing masking contract (arXiv:2605.13779 §6.3) count rides the per-microbatch metrics (reduce_metrics folds it);
    # outside a rollout-replay pass the key is absent.
    engine = _engine(engine_config=_engine_config())
    engine._build_verl_model_output = lambda **_kwargs: {"log_probs": torch.tensor(0.0)}
    engine.get_data_parallel_group = lambda: None
    hook = engine._make_runtime_loss_fn(
        lambda model_output, **_kwargs: (torch.tensor(0.0), {}), 1, output_lst=None
    )

    _, metrics = hook({}, object(), LossContext(source_batch=object()))
    assert "router_replay/unmappable_frac" not in metrics

    engine._router_replay_unmappable_frac = 0.25
    _, metrics = hook({}, object(), LossContext(source_batch=object()))
    assert metrics["router_replay/unmappable_frac"] == 0.25


def test_engine_folds_recorded_routing_to_per_sample_layout() -> None:
    # WS1 §1.3: per-router {mb: [tokens, K]} records fold to per-sample [T_i, L, K]
    # jagged rows in the ORIGINAL batch order via the stashed split layout, for both
    # dynamic-bsz index maps and sequential (indices=None) chunking.
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    engine = MegatronLiteEngine.__new__(MegatronLiteEngine)
    K = 2
    # microbatch 0 holds samples [2, 0] (T=1, 2), microbatch 1 holds sample [1] (T=3)
    recorded = []
    for layer in range(2):
        recorded.append(
            {
                0: torch.arange(3 * K).reshape(3, K) + 100 * layer,
                1: torch.arange(3 * K).reshape(3, K) + 100 * layer + 10,
            }
        )
    engine._router_replay_layout = [
        ([2, 0], torch.tensor([1, 2])),
        ([1], torch.tensor([3])),
    ]

    folded = engine._fold_recorded_routing(recorded)

    rows = list(folded.unbind())
    assert [tuple(r.shape) for r in rows] == [(2, 2, K), (3, 2, K), (1, 2, K)]
    # sample 2 is the FIRST token span of microbatch 0, in every layer
    assert torch.equal(rows[2][:, 0, :], recorded[0][0][:1])
    assert torch.equal(rows[2][:, 1, :], recorded[1][0][:1])
    assert torch.equal(rows[0][:, 0, :], recorded[0][0][1:])
    assert torch.equal(rows[1][:, 1, :], recorded[1][1])

    # sequential fold when the splitter returned no index map
    engine._router_replay_layout = [(None, torch.tensor([1, 2])), (None, torch.tensor([3]))]
    sequential = engine._fold_recorded_routing(recorded)
    assert [tuple(r.shape) for r in sequential.unbind()] == [(1, 2, K), (2, 2, K), (3, 2, K)]

    # incomplete records (a router never ran for microbatch 1) fail loudly
    engine._router_replay_layout = [([0], torch.tensor([3])), ([1], torch.tensor([3]))]
    with pytest.raises(RuntimeError, match="incomplete"):
        engine._fold_recorded_routing([{0: recorded[0][0]}, dict(recorded[1])])


def test_recompute_spec_gates_replay_backward_arming() -> None:
    from verl_mlite.engine.mlite_engine import _recompute_arms_router_replay

    assert _recompute_arms_router_replay({"recompute": "full"}) is True
    assert _recompute_arms_router_replay({"recompute": ["moe"]}) is True
    assert _recompute_arms_router_replay({"recompute": ["router", "core_attn"]}) is True
    # selective attention/expert-only recompute never re-runs the router
    assert _recompute_arms_router_replay({"recompute": ["core_attn", "moe_act"]}) is False
    assert _recompute_arms_router_replay({"recompute": "none"}) is False
    assert _recompute_arms_router_replay({}) is False
    assert _recompute_arms_router_replay(None) is False


def test_engine_export_merges_lora_when_adapter_enabled(monkeypatch) -> None:
    from types import SimpleNamespace

    import verl_mlite.engine.mlite_engine as engine_mod
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    # CUDA cache hygiene is irrelevant to the export-kwargs contract under test
    monkeypatch.setattr(engine_mod, "aggressive_empty_cache", lambda **_: None)

    captured: dict = {}

    class _FakeRuntime:
        @staticmethod
        def export_weights(handle, **kwargs):
            captured.update(kwargs)
            return iter(())

    def make(lora):
        engine = MegatronLiteEngine.__new__(MegatronLiteEngine)
        engine._require_initialized = lambda: None
        # is_param_offload_enabled is a property over engine_config offload flags
        engine.engine_config = SimpleNamespace(
            model_name="qwen2", export_dtype=None, param_offload=False
        )
        engine._mlite_config = SimpleNamespace(impl_cfg={"lora": lora} if lora else {})
        engine.runtime = _FakeRuntime()
        engine.handle = object()
        return engine

    captured.clear()
    make({"rank": 16, "alpha": 32}).get_per_tensor_param()
    assert captured.get("merge_lora") is True

    captured.clear()
    make(None).get_per_tensor_param()
    assert "merge_lora" not in captured

    captured.clear()
    make({"rank": 0}).get_per_tensor_param()
    assert "merge_lora" not in captured
