#!/usr/bin/env python3
"""Run correctness while fingerprinting Qwen3.5 MRoPE generation/application."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import torch


def _tensor(value):
    if not isinstance(value, torch.Tensor):
        return None
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _install_mlite() -> None:
    import megatron.lite.primitive.modules.gqa as gqa
    from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding

    original_freq = MultimodalRotaryEmbedding.forward

    def freq(self, position_ids, mrope_section, **kwargs):
        output = original_freq(self, position_ids, mrope_section, **kwargs)
        print(
            "ROPE_FREQ "
            + json.dumps(
                {
                    "impl": "mlite",
                    "position_ids": _tensor(position_ids),
                    "section": list(mrope_section),
                    "inv_freq": _tensor(self.inv_freq),
                    "output": _tensor(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return output

    MultimodalRotaryEmbedding.forward = freq
    original_apply = gqa._apply_rotary_pos_emb_bshd

    def apply(t, freqs, *args, **kwargs):
        output = original_apply(t, freqs, *args, **kwargs)
        print(
            "ROPE_APPLY "
            + json.dumps(
                {
                    "impl": "mlite",
                    "t": _tensor(t),
                    "freqs": _tensor(freqs),
                    "args": [repr(item) for item in args],
                    "kwargs": {key: repr(value) for key, value in sorted(kwargs.items())},
                    "output": _tensor(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return output

    gqa._apply_rotary_pos_emb_bshd = apply


def _install_mbridge() -> None:
    import mbridge.models.qwen3_5.attention as attention
    from mbridge.models.qwen3_vl.rope_utils import Qwen3VLMultimodalRotaryEmbedding

    original_freq = Qwen3VLMultimodalRotaryEmbedding.forward

    def freq(self, position_ids, mrope_section, **kwargs):
        output = original_freq(self, position_ids, mrope_section, **kwargs)
        print(
            "ROPE_FREQ "
            + json.dumps(
                {
                    "impl": "mbridge",
                    "position_ids": _tensor(position_ids),
                    "section": list(mrope_section),
                    "inv_freq": _tensor(self.inv_freq),
                    "output": _tensor(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return output

    Qwen3VLMultimodalRotaryEmbedding.forward = freq
    original_apply = attention.apply_rotary_pos_emb_absolute

    def apply(t, freqs, config, *args, **kwargs):
        output = original_apply(t, freqs, config, *args, **kwargs)
        print(
            "ROPE_APPLY "
            + json.dumps(
                {
                    "impl": "mbridge",
                    "fp32": bool(config.apply_rotary_pos_emb_in_fp32),
                    "t": _tensor(t),
                    "freqs": _tensor(freqs),
                    "args": [repr(item) for item in args],
                    "kwargs": {key: repr(value) for key, value in sorted(kwargs.items())},
                    "output": _tensor(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return output

    attention.apply_rotary_pos_emb_absolute = apply


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: rope_call_probe.py CORRECTNESS_PY [ARGS ...]")
    target = Path(sys.argv[1]).resolve()
    sys.argv = [str(target), *sys.argv[2:]]
    _install_mlite()
    _install_mbridge()
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
