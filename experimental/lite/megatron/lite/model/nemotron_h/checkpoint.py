"""Explicit HF tensor mapping for locally owned native Nemotron parameters."""

import json
from pathlib import Path

import torch


class NemotronExport:
    """Expose stacked expert views to the framework's EP/PP exporter."""

    def __init__(self, config):
        self.num_experts = config.n_routed_experts

    @staticmethod
    def is_expert(name):
        return ".mixer.experts." in name

    @staticmethod
    def export_expert_local_id(name):
        return int(name.split(".experts.", 1)[1].split(".", 1)[0])

    @staticmethod
    def export_expert_name(name, index):
        prefix, suffix = name.split(".experts.", 1)
        return f"{prefix}.experts.{index}.{suffix.split('.', 1)[1]}"

    @staticmethod
    def tp_spec(name):
        return None

    @staticmethod
    def native_to_hf(name, tensor):
        return [(name, tensor)]

    def iter_export_tensors(self, model):
        first = model.ps.ep_rank * (self.num_experts // model.ps.ep_size)
        for name, tensor in hf_tensor_views(model):
            if self.is_expert(name):
                name = self.export_expert_name(
                    name, self.export_expert_local_id(name) - first
                )
            yield name, tensor.detach()


def hf_tensor_views(model):
    """Yield HF names and destination views, without gathering remote experts."""
    local_experts = model.config.n_routed_experts // model.ps.ep_size
    first_expert = model.ps.ep_rank * local_experts
    for name, tensor in model.state_dict(keep_vars=True).items():
        if ".mixer.experts." in name:
            prefix, projection = name.rsplit(".", 1)
            if projection not in ("up_proj", "down_proj"):
                raise ValueError(f"Unrecognized expert weight: {name}")
            if tensor.shape[0] != local_experts:
                raise ValueError(f"Incorrect local expert ownership: {name}")
            for local in range(local_experts):
                yield (
                    f"backbone.{prefix}.{first_expert + local}.{projection}.weight",
                    tensor[local],
                )
        else:
            yield (name if name.startswith("lm_head.") else f"backbone.{name}"), tensor


@torch.no_grad()
def load_hf_weights(model, path):
    """Strictly load local tensors; unrelated PP/EP shards stay on disk."""
    from safetensors import safe_open

    root = Path(path)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    targets = dict(hf_tensor_views(model))
    missing = targets.keys() - index.keys()
    if missing:
        raise ValueError(f"Missing required HF weights: {sorted(missing)}")
    for filename in sorted({index[name] for name in targets}):
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for name, target in targets.items():
                if index[name] != filename:
                    continue
                value = handle.get_tensor(name)
                if value.shape != target.shape:
                    raise ValueError(
                        f"HF shape mismatch for {name}: {value.shape} != {target.shape}"
                    )
                if target.is_meta:
                    raise ValueError(
                        "Materialize model parameters before loading weights"
                    )
                target.copy_(value)
