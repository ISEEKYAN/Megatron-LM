# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Conformance suite for the absorbing composition kernel.

These tests run on CPU (no CUDA / transformer_engine): they drive
``compose.assemble`` with a fake :class:`ModelSpec` and fake chunks, so
they verify the *shared assembly contract* — chunk construction, the single
``transformer_units`` enumeration (#114), optimizer wiring, and ``ModelBundle``
packing — without building a real model.

They also assert the migrated DeepSeek-V4 protocol declares a spec and delegates
to the kernel rather than re-implementing the body.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

LITE_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = LITE_ROOT / "megatron" / "lite" / "model"
#: Every native model whose build_model must delegate to compose.assemble.
MIGRATED_MODELS = ["deepseek_v4", "glm5", "kimi_k2", "qwen3_5", "qwen3_moe"]
DS4_PROTOCOL = MODEL_ROOT / "deepseek_v4" / "lite" / "protocol.py"


class _FakeChunk:
    """Stands in for a built model chunk. ``.to().cuda()`` are no-ops."""

    def __init__(self, chunk_id):
        self.chunk_id = chunk_id
        self.units = [SimpleNamespace(name=f"unit{chunk_id}.0"), SimpleNamespace(name=f"unit{chunk_id}.1")]

    def to(self, *_a, **_k):
        return self

    def cuda(self):
        return self


def _fake_impl_cfg(**overrides):
    base = dict(
        parallel=SimpleNamespace(vpp=1, tp=1, ep=1, cp=1, etp=None),
        optimizer=None,
        optimizer_config=None,
        recompute=[],
        offload=[],
        deterministic=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_spec(monkeypatch, **spec_overrides):
    from megatron.lite.model import compose
    from megatron.lite.primitive.parallel import ParallelState

    # Keep assemble on CPU: no real parallel init, no CUDA.
    monkeypatch.setattr(compose, "init_parallel", lambda _cfg: ParallelState())

    events = {"recompute": [], "offload": [], "prepare": [], "post_chunk": []}

    def chunk_factory(model_cfg, impl_cfg, ps, vpp_chunk_id):
        return _FakeChunk(vpp_chunk_id if vpp_chunk_id is not None else "single")

    spec_kwargs = dict(
        name="fake_model",
        chunk_factory=chunk_factory,
        transformer_units=lambda chunk: chunk.units,
        module_map={},
        forward_step=lambda model, batch: {},
        expert_classifier=lambda name: False,
        placement_fn=lambda *a, **k: None,
        fsdp2_unit_modules=lambda: (),
    )
    spec_kwargs.update(spec_overrides)
    spec = compose.ModelSpec(**spec_kwargs)
    return compose, spec, events


def test_assemble_single_chunk_bundle(monkeypatch):
    compose, spec, _ = _make_spec(monkeypatch)
    fwd = spec.forward_step
    bundle = compose.assemble(spec, model_cfg=SimpleNamespace(), impl_cfg=_fake_impl_cfg())

    assert len(bundle.chunks) == 1
    assert bundle.forward_step is fwd
    assert bundle.optimizer is None
    assert bundle.finalize_grads is None
    assert bundle.extras["optimizer_backend"] == "none"
    assert bundle.extras["post_model_load_hook"] is None
    # No aux-loss factory declared -> no pre_forward_hook key.
    assert "pre_forward_hook" not in bundle.extras


def test_assemble_vpp_builds_one_chunk_per_stage(monkeypatch):
    compose, spec, _ = _make_spec(monkeypatch)
    impl = _fake_impl_cfg(parallel=SimpleNamespace(vpp=3, tp=1, ep=1, cp=1, etp=None))
    bundle = compose.assemble(spec, model_cfg=SimpleNamespace(), impl_cfg=impl)
    assert [c.chunk_id for c in bundle.chunks] == [0, 1, 2]


def test_prepare_and_post_chunk_hooks_ordering(monkeypatch):
    compose, _, _ = _make_spec(monkeypatch)
    order = []

    compose_mod, spec, _ = _make_spec(
        monkeypatch,
        prepare=lambda cfg, impl, ps: order.append("prepare"),
        post_chunk_hook=lambda chunks, impl: order.append("post_chunk"),
    )
    # chunk build happens between prepare and post_chunk_hook
    orig_factory = spec.chunk_factory

    def tracking_factory(*a, **k):
        order.append("chunk")
        return orig_factory(*a, **k)

    spec = compose_mod.ModelSpec(
        **{**{f.name: getattr(spec, f.name) for f in spec.__dataclass_fields__.values()},
           "chunk_factory": tracking_factory}
    )
    compose_mod.assemble(spec, model_cfg=SimpleNamespace(), impl_cfg=_fake_impl_cfg())
    assert order == ["prepare", "chunk", "post_chunk"]


def test_recompute_offload_go_through_transformer_units(monkeypatch):
    """#114: recompute/offload must walk the single ``transformer_units`` list."""
    compose, spec, _ = _make_spec(monkeypatch)
    seen = {"recompute": [], "offload": []}

    monkeypatch.setattr(
        compose,
        "parse_recompute_spec",
        lambda names: list(names),
    )
    monkeypatch.setattr(
        compose,
        "apply_recompute",
        lambda units, names, mmap: seen["recompute"].append((list(units), names)),
    )
    monkeypatch.setattr(
        compose,
        "apply_offload",
        lambda units, names, mmap: seen["offload"].append((list(units), names)),
    )

    impl = _fake_impl_cfg(recompute=["full"], offload=["moe"])
    bundle = compose.assemble(spec, model_cfg=SimpleNamespace(), impl_cfg=impl)

    # Both passes received exactly the units yielded by spec.transformer_units.
    expected_units = spec.transformer_units(bundle.chunks[0])
    assert seen["recompute"] == [(list(expected_units), ["full"])]
    assert seen["offload"] == [(list(expected_units), ["moe"])]


def test_recompute_offload_skipped_when_empty(monkeypatch):
    compose, spec, _ = _make_spec(monkeypatch)
    called = {"recompute": 0, "offload": 0}
    monkeypatch.setattr(compose, "parse_recompute_spec", lambda names: [])
    monkeypatch.setattr(
        compose, "apply_recompute", lambda *a: called.__setitem__("recompute", called["recompute"] + 1)
    )
    monkeypatch.setattr(
        compose, "apply_offload", lambda *a: called.__setitem__("offload", called["offload"] + 1)
    )
    compose.assemble(spec, model_cfg=SimpleNamespace(), impl_cfg=_fake_impl_cfg())
    assert called == {"recompute": 0, "offload": 0}


def test_optimizer_backend_name_normalizer(monkeypatch):
    compose, _, _ = _make_spec(monkeypatch)
    _, spec, _ = _make_spec(
        monkeypatch,
        optimizer_backend_name=lambda opt: "dist_opt" if isinstance(opt, dict) else opt,
    )
    # A dict optimizer must be normalized to dist_opt (would otherwise raise).
    monkeypatch.setattr(
        compose,
        "_wire_optimizer",
        lambda *a, **k: (object(), lambda: None, None, "dist_opt"),
    )
    bundle = compose.assemble(
        spec, model_cfg=SimpleNamespace(), impl_cfg=_fake_impl_cfg(optimizer={"lr": 1.0})
    )
    assert bundle.extras["optimizer_backend"] == "dist_opt"


def test_wire_optimizer_applies_normalizer_before_dispatch(monkeypatch):
    """DS4 passes an OptimizerConfig/dict as ``optimizer``; the normalizer must
    map it to ``dist_opt`` *inside* _wire_optimizer, before the backend switch."""
    from megatron.lite.model import compose
    from megatron.lite.primitive.parallel import ParallelState

    _, spec, _ = _make_spec(
        monkeypatch,
        optimizer_backend_name=lambda opt: "dist_opt" if isinstance(opt, dict) else opt,
    )
    # Stub the dist_opt wiring internals so we don't touch CUDA/megatron.
    import types

    fake_megatron_wrap = types.ModuleType("megatron.lite.primitive.optimizers.megatron_wrap")
    fake_megatron_wrap.build_dist_opt_training_optimizer = lambda *a, **k: ("OPT", "FINALIZE")
    fake_ckpt = types.ModuleType("megatron.lite.primitive.ckpt")
    fake_ckpt.attach_model_sharded_state_dict = lambda *a, **k: None
    fake_mutils = types.ModuleType("megatron.lite.runtime.megatron_utils")
    fake_mutils.register_training_hooks = lambda *a, **k: None
    monkeypatch.setitem(
        __import__("sys").modules,
        "megatron.lite.primitive.optimizers.megatron_wrap",
        fake_megatron_wrap,
    )
    monkeypatch.setitem(__import__("sys").modules, "megatron.lite.primitive.ckpt", fake_ckpt)
    monkeypatch.setitem(
        __import__("sys").modules, "megatron.lite.runtime.megatron_utils", fake_mutils
    )

    impl = _fake_impl_cfg(optimizer={"lr": 1.0})
    opt, finalize, hook, backend = compose._wire_optimizer(
        spec, chunks=[_FakeChunk("c")], model_cfg=SimpleNamespace(), impl_cfg=impl, ps=ParallelState()
    )
    assert (opt, finalize, hook, backend) == ("OPT", "FINALIZE", None, "dist_opt")


def test_unknown_optimizer_raises(monkeypatch):
    compose, spec, _ = _make_spec(monkeypatch)
    with pytest.raises(ValueError, match="Unknown fake_model lite optimizer"):
        compose.assemble(
            spec, model_cfg=SimpleNamespace(), impl_cfg=_fake_impl_cfg(optimizer="bogus")
        )


def test_pre_forward_hook_factory_populates_extras(monkeypatch):
    sentinel = object()
    compose, spec, _ = _make_spec(
        monkeypatch, pre_forward_hook_factory=lambda: sentinel
    )
    bundle = compose.assemble(spec, model_cfg=SimpleNamespace(), impl_cfg=_fake_impl_cfg())
    assert bundle.extras["pre_forward_hook"] is sentinel


def test_extra_extras_merge_into_bundle(monkeypatch):
    compose, spec, _ = _make_spec(
        monkeypatch, extra_extras=lambda chunks, impl: {"lora_config": "L", "lora_stats": None}
    )
    bundle = compose.assemble(spec, model_cfg=SimpleNamespace(), impl_cfg=_fake_impl_cfg())
    assert bundle.extras["lora_config"] == "L"
    assert bundle.extras["lora_stats"] is None


def test_extra_extras_run_before_optimizer(monkeypatch):
    """qwen3_moe LoRA must freeze params (extra_extras) before the optimizer sees them."""
    compose, _, _ = _make_spec(monkeypatch)
    order = []
    monkeypatch.setattr(
        compose, "_wire_optimizer", lambda *a, **k: order.append("optimizer") or (None, None, None, "none")
    )
    _, spec, _ = _make_spec(
        monkeypatch, extra_extras=lambda chunks, impl: order.append("extra_extras") or {}
    )
    monkeypatch.setattr(
        compose, "_wire_optimizer", lambda *a, **k: order.append("optimizer") or (None, None, None, "none")
    )
    compose.assemble(spec, model_cfg=SimpleNamespace(), impl_cfg=_fake_impl_cfg())
    assert order == ["extra_extras", "optimizer"]


def test_register_hooks_false_skips_training_hooks(monkeypatch):
    """qwen3_moe sets register_hooks=False; the dist_opt path must not register them."""
    from megatron.lite.primitive.parallel import ParallelState
    import types

    _, spec, _ = _make_spec(monkeypatch, register_hooks=False)
    calls = {"register": 0}
    fake_mw = types.ModuleType("megatron.lite.primitive.optimizers.megatron_wrap")
    fake_mw.build_dist_opt_training_optimizer = lambda *a, **k: ("OPT", "FIN")
    fake_ckpt = types.ModuleType("megatron.lite.primitive.ckpt")
    fake_ckpt.attach_model_sharded_state_dict = lambda *a, **k: None
    fake_mu = types.ModuleType("megatron.lite.runtime.megatron_utils")
    fake_mu.register_training_hooks = lambda *a, **k: calls.__setitem__("register", calls["register"] + 1)
    import sys
    monkeypatch.setitem(sys.modules, "megatron.lite.primitive.optimizers.megatron_wrap", fake_mw)
    monkeypatch.setitem(sys.modules, "megatron.lite.primitive.ckpt", fake_ckpt)
    monkeypatch.setitem(sys.modules, "megatron.lite.runtime.megatron_utils", fake_mu)
    from megatron.lite.model import compose
    compose._wire_optimizer(
        spec, chunks=[_FakeChunk("c")], model_cfg=SimpleNamespace(),
        impl_cfg=_fake_impl_cfg(optimizer="dist_opt"), ps=ParallelState()
    )
    assert calls["register"] == 0


def test_fsdp2_deterministic_fn_overrides(monkeypatch):
    """qwen3_5 forces deterministic=False for THD GatedDeltaNet fsdp2 wiring."""
    from megatron.lite.primitive.parallel import ParallelState
    import types, sys

    _, spec, _ = _make_spec(monkeypatch, fsdp2_deterministic_fn=lambda impl: False)
    seen = {}
    fake_fsdp2 = types.ModuleType("megatron.lite.primitive.optimizers.fsdp2")
    fake_fsdp2.build_fsdp2_training_optimizer = lambda *a, **k: seen.update(k) or "OPT"
    monkeypatch.setitem(sys.modules, "megatron.lite.primitive.optimizers.fsdp2", fake_fsdp2)
    from megatron.lite.model import compose
    _, _, hook, backend = compose._wire_optimizer(
        spec, chunks=[_FakeChunk("c")], model_cfg=SimpleNamespace(),
        impl_cfg=_fake_impl_cfg(optimizer="fsdp2", parallel=SimpleNamespace(vpp=1)),
        ps=ParallelState(),
    )
    assert backend == "fsdp2"
    hook()  # trigger the lazy fsdp2 build
    assert seen["deterministic"] is False


# --------------------------------------------------------------------------
# DS4 migration conformance: the protocol declares a spec + calls assemble.
# --------------------------------------------------------------------------


def _build_model_ast(model: str) -> ast.FunctionDef:
    protocol = MODEL_ROOT / model / "lite" / "protocol.py"
    tree = ast.parse(protocol.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_model":
            return node
    raise AssertionError(f"{model} protocol has no build_model")


@pytest.mark.parametrize("model", MIGRATED_MODELS)
def test_build_model_delegates_to_kernel(model):
    fn = _build_model_ast(model)
    calls = {
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "ModelSpec" in calls, f"{model} build_model must declare a ModelSpec"
    assert "assemble" in calls, f"{model} build_model must delegate to assemble"


@pytest.mark.parametrize("model", MIGRATED_MODELS)
def test_build_model_has_no_inline_assembly(model):
    """Migrated build_model must not re-implement the shared body."""
    fn = _build_model_ast(model)
    banned = {"init_parallel", "apply_recompute", "apply_offload", "ModelBundle"}
    names = {
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    leaked = banned & names
    assert not leaked, f"{model} build_model still inlines shared assembly: {leaked}"


@pytest.mark.parametrize("model", MIGRATED_MODELS)
def test_build_model_is_small(model):
    """Net-negative proof: the delegating build_model is a thin declaration."""
    fn = _build_model_ast(model)
    stmt_lines = {
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.stmt) and node is not fn
    }
    assert len(stmt_lines) < 20, (
        f"{model} build_model should be a thin delegator, got {len(stmt_lines)} statement lines"
    )
