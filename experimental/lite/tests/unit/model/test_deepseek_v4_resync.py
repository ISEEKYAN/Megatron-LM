import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


_RESYNC_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron"
    / "lite"
    / "model"
    / "deepseek_v4"
    / "lite"
    / "resync.py"
)
_SPEC = importlib.util.spec_from_file_location("deepseek_v4_resync_under_test", _RESYNC_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
export_resync_weights = _MODULE.export_resync_weights


def test_fp8_resync_exports_float32_block_scales() -> None:
    config = SimpleNamespace(
        expert_dtype="fp8",
        quantization_config={"weight_block_size": [128, 128]},
    )
    source = torch.randn(128, 128, dtype=torch.bfloat16)

    exported = dict(
        export_resync_weights(
            [("layers.0.ffn.experts.0.up_proj.weight", source)],
            config,
            resync_config={"expert_dtype": "fp8"},
        )
    )

    assert exported["layers.0.ffn.experts.0.up_proj.scale"].dtype == torch.float32
