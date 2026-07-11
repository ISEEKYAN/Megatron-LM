# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

LITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
VERL_EXAMPLE_ROOT = LITE_ROOT / "examples" / "verl"
for root in (REPO_ROOT, LITE_ROOT, VERL_EXAMPLE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def pytest_configure(config):
    config.addinivalue_line("markers", "mlite: mark a test as Megatron Lite validation coverage")
    config.addinivalue_line(
        "markers",
        "smoke: mark a Megatron Lite smoke test; skipped unless --mlite-smoke or MLITE_RUN_SMOKE=1 is set",
    )
    config.addinivalue_line("markers", "gpu: mark a test as requiring CUDA")
    config.addinivalue_line("markers", "distributed: mark a test as requiring torch.distributed")


def pytest_addoption(parser):
    parser.addoption(
        "--mlite-smoke", action="store_true", default=False, help="run Megatron Lite smoke tests"
    )


def pytest_collection_modifyitems(config, items):
    run_smoke = config.getoption("--mlite-smoke") or os.getenv("MLITE_RUN_SMOKE") == "1"
    if run_smoke:
        return
    skip_smoke = pytest.mark.skip(reason="set --mlite-smoke or MLITE_RUN_SMOKE=1 to run")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip_smoke)


def _build_transformer_engine_stub_modules() -> dict[str, types.ModuleType]:
    class _UnavailableTE:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Transformer Engine is not installed in this test environment.")

    root = types.ModuleType("transformer_engine")
    root.__version__ = "0.0.0"
    pytorch = types.ModuleType("transformer_engine.pytorch")
    pytorch.DotProductAttention = _UnavailableTE
    pytorch.LayerNormLinear = _UnavailableTE
    pytorch.Linear = _UnavailableTE
    pytorch.GroupedLinear = _UnavailableTE
    pytorch.RMSNorm = _UnavailableTE
    permutation = types.ModuleType("transformer_engine.pytorch.permutation")
    router = types.ModuleType("transformer_engine.pytorch.router")
    cpp_extensions = types.ModuleType("transformer_engine.pytorch.cpp_extensions")
    module = types.ModuleType("transformer_engine.pytorch.module")
    module_base = types.ModuleType("transformer_engine.pytorch.module.base")

    def unavailable_kernel(*args, **kwargs):
        raise RuntimeError("Transformer Engine fused kernel is not installed.")

    permutation.moe_permute = unavailable_kernel
    permutation.moe_permute_and_pad_with_probs = unavailable_kernel
    permutation.moe_permute_with_probs = unavailable_kernel
    permutation.moe_unpermute = unavailable_kernel
    router.fused_compute_score_for_moe_aux_loss = unavailable_kernel
    router.fused_moe_aux_loss = unavailable_kernel
    router.fused_topk_with_score_function = unavailable_kernel
    cpp_extensions.general_gemm = lambda *args, **kwargs: None
    module_base.get_workspace = lambda: None
    module.base = module_base
    pytorch.permutation = permutation
    pytorch.router = router
    pytorch.cpp_extensions = cpp_extensions
    pytorch.module = module
    root.pytorch = pytorch
    return {
        "transformer_engine": root,
        "transformer_engine.pytorch": pytorch,
        "transformer_engine.pytorch.permutation": permutation,
        "transformer_engine.pytorch.router": router,
        "transformer_engine.pytorch.cpp_extensions": cpp_extensions,
        "transformer_engine.pytorch.module": module,
        "transformer_engine.pytorch.module.base": module_base,
    }


# Built once and reused across tests so that the module *identity* stays stable.
# Primitive modules cache ``import transformer_engine.pytorch as te`` at import
# time, while callers such as PrecisionCoverage resolve it lazily; sharing one
# stub object keeps those references in agreement even though ``monkeypatch``
# reverts the sys.modules entries between tests.
_TE_STUB_MODULES: dict[str, types.ModuleType] | None = None


@pytest.fixture
def transformer_engine_import_stub(monkeypatch):
    def install() -> None:
        try:
            import transformer_engine.pytorch  # noqa: F401

            return
        except (ModuleNotFoundError, OSError) as exc:
            if isinstance(exc, ModuleNotFoundError) and exc.name not in {
                "transformer_engine",
                "transformer_engine.pytorch",
            }:
                raise

        global _TE_STUB_MODULES
        if _TE_STUB_MODULES is None:
            _TE_STUB_MODULES = _build_transformer_engine_stub_modules()
        for name, mod in _TE_STUB_MODULES.items():
            monkeypatch.setitem(sys.modules, name, mod)

    return install
