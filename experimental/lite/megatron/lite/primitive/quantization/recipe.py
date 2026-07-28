# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""vLLM-compatible QAT recipes and a non-invasive prepare/export boundary."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from os import PathLike

import torch
import torch.nn as nn


class QATRecipe(str, Enum):
    INT4_W4A16_G128 = "int4_w4a16_g128"
    NVFP4_W4A16 = "nvfp4_w4a16"


@dataclass(frozen=True)
class RecipeContract:
    weight_format: str
    group_size: int
    scale_format: str
    activation_quantized: bool
    compressed_tensors_scheme: str


_CONTRACTS = {
    QATRecipe.INT4_W4A16_G128: RecipeContract("int4", 128, "float", False, "W4A16"),
    QATRecipe.NVFP4_W4A16: RecipeContract("float4_e2m1", 16, "float8_e4m3fn", False, "NVFP4A16"),
}


def recipe_contract(recipe: QATRecipe | str) -> RecipeContract:
    """Return the frozen ModelOpt/CT/vLLM contract; INT4 g16 is intentionally absent."""
    try:
        return _CONTRACTS[QATRecipe(recipe)]
    except ValueError as exc:
        raise ValueError(f"Unsupported QAT recipe {recipe!r}; use {list(QATRecipe)}.") from exc


class QuantizerB:
    """torchao-shaped two-stage API without adding torchao to the runtime.

    ModelOpt owns fake quantization in ``prepare`` and its unified-HF exporter
    owns the deployment artifact.  A ModelOpt Q/DQ hierarchy is not a
    compressed-tensors hierarchy: CT cannot infer the latter's state from the
    former, so this API must never silently pass BF16 tensors through as an
    allegedly quantized checkpoint.
    """

    def __init__(self, recipe: QATRecipe | str):
        self.recipe = QATRecipe(recipe)
        self.contract = recipe_contract(self.recipe)

    def prepare(
        self,
        model: nn.Module,
        calibration_forward_loop: Callable[[nn.Module], None] | None = None,
    ) -> nn.Module:
        """Insert ModelOpt Q/DQ into one complete model.

        ``mtq.quantize`` transforms a complete module hierarchy in place.  It
        is deliberately not called one leaf at a time: ModelOpt registers its
        quantizer subclasses while traversing the owning model and optional
        activation calibration must execute through that same hierarchy.
        INT4 W4A16 uses ModelOpt's weight-only/max preset, so its caller may
        omit ``calibration_forward_loop``.
        """
        try:
            import modelopt.torch.quantization as mtq
        except ImportError as exc:
            raise RuntimeError(
                "QAT recipe preparation requires NVIDIA ModelOpt; do not fall back to MLite fake quant."
            ) from exc
        return mtq.quantize(model, self._modelopt_config(), calibration_forward_loop)

    def export(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Reject the former tensor-iterator pseudo-export.

        Kept as a narrow compatibility failure rather than returning the input:
        an unchanged BF16 iterator is not a quantized vLLM artifact.
        """
        del weights
        raise RuntimeError(
            "QAT recipe export requires export_hf_checkpoint(model, output_dir); "
            "ModelOpt Q/DQ cannot be CT-packed by tensor passthrough."
        )
        yield  # pragma: no cover - retains the historical iterator annotation.

    def export_hf_checkpoint(self, model: nn.Module, output_dir: str | PathLike[str]) -> None:
        """Write ModelOpt's quantized unified-HF artifact for vLLM loading."""
        try:
            from modelopt.torch.export import export_hf_checkpoint
        except ImportError as exc:
            raise RuntimeError("QAT recipe export requires NVIDIA ModelOpt's unified-HF exporter.") from exc
        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=str(output_dir))

    def _modelopt_config(self) -> dict:
        """Copy a ModelOpt-owned preset instead of encoding its private schema."""
        try:
            import modelopt.torch.quantization as mtq
        except ImportError as exc:
            raise RuntimeError(
                "QAT recipe preparation requires NVIDIA ModelOpt; do not fall back to MLite fake quant."
            ) from exc
        if self.recipe is QATRecipe.INT4_W4A16_G128:
            return copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
        # ModelOpt 0.43 names its public W4A16 NVFP4 preset
        # ``NVFP4_DEFAULT_CFG``.  Do not use the later W4A16_NVFP4_CFG alias:
        # the pinned runtime intentionally remains 0.43 because 0.44/0.45 have
        # a registry failure in this container.
        return copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)


__all__ = ["QATRecipe", "QuantizerB", "RecipeContract", "recipe_contract"]
