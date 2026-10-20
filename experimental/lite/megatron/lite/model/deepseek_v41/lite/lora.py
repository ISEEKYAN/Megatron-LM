# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from megatron.lite.primitive.modules.lora import (
    LinearLoRA,
    LoraConfig,
    lora_active,
    register_lora_adapter_type,
)
from megatron.lite.primitive.modules.moe_lora import SharedOuterExpertLoRA, expand_shared_outer

_ATTENTION = {
    "wq_a": "linear_q_down_proj",
    "wq_b": "linear_q_up_proj",
    "wkv": "linear_kv_proj",
    "wo_b": "linear_proj",
}
TARGETS = (*_ATTENTION, "wo_a", "shared_experts", "experts")


class GroupedLoRA(nn.Module):

    def __init__(self, groups, in_features, out_features, rank, scale):
        super().__init__()
        self.disabled = False
        self.scale = scale
        self.lora_a = nn.Parameter(torch.empty(groups, rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(groups, out_features, rank))
        for a in self.lora_a.data:
            nn.init.kaiming_uniform_(a, a=5**0.5)

    def forward(self, x):
        x = x.unflatten(-1, (self.lora_a.shape[0], -1))
        hidden = torch.einsum("...gd,grd->...gr", x, self.lora_a)
        return (torch.einsum("...gr,gor->...go", hidden, self.lora_b) * self.scale).flatten(-2)


def lora_layers(cfg, num_layers):
    start, end = cfg.layers or (max(0, num_layers - 20), num_layers)
    if not 0 <= start < end <= num_layers:
        raise ValueError(f"DS4.1 LoRA layers [{start}, {end}) outside [0, {num_layers})")
    return start, end


def attach_lora(model, cfg, dtype):
    targets = set(
        TARGETS if cfg.target_modules == LoraConfig().target_modules else cfg.target_modules
    )
    if targets - set(TARGETS) or cfg.dropout:
        raise ValueError(f"DS4.1 LoRA supports targets {TARGETS} without dropout")
    start, end = lora_layers(cfg, model.release.num_hidden_layers)
    with torch.random.fork_rng(devices=[]):  # base initialization stays unchanged
        for key, block in model.layers.items():
            if not start <= int(key) < end:
                continue
            attention, shared = block.attention, block.ffn.shared_experts
            for name, base in _ATTENTION.items():
                if name in targets:
                    _add(attention, name, getattr(attention, base), cfg, dtype)
            if "wo_a" in targets:
                groups = attention.o_local_groups
                rank_out, width = attention.linear_o_group_proj.shape
                lora = GroupedLoRA(groups, width, rank_out // groups, cfg.rank, cfg.scale)
                attention.wo_a_lora = lora.to(dtype)
                attention.core_attention.register_forward_hook(
                    lambda m, args, out, a=attention: (
                        setattr(a, "_wo_a_input", out) if lora_active(a.wo_a_lora) else None
                    )
                )
                attention.linear_proj.register_forward_pre_hook(
                    lambda m, args, a=attention: (
                        args
                        if not lora_active(a.wo_a_lora)
                        else (
                            args[0]
                            + a.wo_a_lora(
                                a.__dict__.pop("_wo_a_input").reshape(*args[0].shape[:-1], -1)
                            ),
                        )
                        + args[1:]
                    )
                )
            if "shared_experts" in targets:
                out, width = shared.linear_fc1.weight.shape
                pair = [LinearLoRA(width, out // 2, cfg.rank, alpha=cfg.alpha) for _ in "13"]
                shared.w1_lora, shared.w3_lora = (p.to(dtype) for p in pair)
                shared.linear_fc1.register_forward_hook(
                    lambda m, args, out, s=shared: (
                        out
                        if not (lora_active(s.w1_lora) or lora_active(s.w3_lora))
                        else (
                            out[0]
                            + torch.cat(
                                tuple(
                                    (
                                        f(args[0])
                                        if lora_active(f)
                                        else out[0].new_zeros(
                                            (*out[0].shape[:-1], out[0].shape[-1] // 2)
                                        )
                                    )
                                    for f in (s.w1_lora, s.w3_lora)
                                ),
                                -1,
                            ),
                            *out[1:],
                        )
                    )
                )
                _add(shared, "w2", shared.linear_fc2, cfg, dtype)
            if "experts" in targets:
                experts, release = block.ffn.experts, model.release
                SharedOuterExpertLoRA(
                    experts.num_local_experts,
                    release.hidden_size,
                    release.moe_intermediate_size,
                    cfg.rank,
                    cfg.alpha or cfg.rank,
                    config=experts.config,
                    groups=model.context.groups,
                ).attach(experts)


def _add(owner, name, base, cfg, dtype):
    out, width = base.weight.shape
    lora = LinearLoRA(width, out, cfg.rank, alpha=cfg.alpha).to(dtype)
    setattr(owner, f"{name}_lora", lora)
    base.register_forward_hook(
        lambda m, args, out, f=lora: out if not lora_active(f) else (out[0] + f(args[0]), *out[1:])
    )


def lora_state(chunk):
    while hasattr(chunk, "module"):
        chunk = chunk.module
    state = {}
    for key, block in chunk.layers.items():
        shared = block.ffn.shared_experts
        owners = [(f"attn.{n}", getattr(block.attention, f"{n}_lora", None)) for n in TARGETS[:5]]
        owners += [
            (f"ffn.shared_experts.{w}", getattr(shared, f"{w}_lora", None))
            for w in "w1 w3 w2".split()
        ]
        for owner, module in owners:
            if module is not None:
                state[f"layers.{key}.{owner}.lora_A.weight"] = module.lora_a
                state[f"layers.{key}.{owner}.lora_B.weight"] = module.lora_b
        experts = getattr(block.ffn.experts, "lora", None)
        if experts is not None:
            state.update({f"layers.{key}.ffn.experts.{k}": v for k, v in experts.factors().items()})
    return state


def serving_lora_state(state, num_experts, scale):
    out = {k: v for k, v in state.items() if ".ffn.experts." not in k}
    for stem in sorted({k.split(".lora_gate_up_A")[0] for k in state if "lora_gate_up_A" in k}):
        stack = lambda f: torch.stack(
            [state[f"{stem}.{e}.lora_{f}.weight"] for e in range(num_experts)]
        )
        expanded = expand_shared_outer(
            state[f"{stem}.lora_gate_up_A.weight"],
            stack("gate_up_B"),
            stack("down_A"),
            state[f"{stem}.lora_down_B.weight"],
            scale,
        )
        out.update({f"{stem}.{k}.weight": v for k, v in expanded.items()})
    return out


def save_lora_hf(chunks, path, cfg):
    from safetensors.torch import save_file

    from megatron.lite.model.deepseek_v41.lite.export import export_hf_weights

    state = dict(export_hf_weights(chunks, None, None, target="lora"))
    if dist.get_rank() == 0:
        Path(path).mkdir(parents=True, exist_ok=True)
        save_file(state, str(Path(path) / "adapter_model.safetensors"))
        layers = sorted({int(k.split(".")[1]) for k in state})
        config = dict(peft_type="LORA", r=cfg.rank, lora_alpha=cfg.alpha or cfg.rank)
        targets = (
            TARGETS if cfg.target_modules == LoraConfig().target_modules else cfg.target_modules
        )
        config.update(target_modules=sorted(set(targets)), layers_to_transform=layers)
        (Path(path) / "adapter_config.json").write_text(json.dumps(config, indent=1))
    dist.barrier()


@torch.no_grad()
def load_lora_hf(chunks, path):
    from safetensors.torch import load_file

    saved = load_file(str(Path(path) / "adapter_model.safetensors"))
    for chunk in chunks:
        for key, tensor in lora_state(chunk).items():
            if key not in saved or saved[key].shape != tensor.shape:
                raise ValueError(f"DS4.1 LoRA adapter file lacks or misshapes {key}")
            tensor.copy_(saved[key])


def _lora_sharded(chunks, ps):
    from megatron.lite.primitive.ckpt.row_state import attach_checkpoint_metadata

    state = {}
    for chunk in chunks:
        if not getattr(chunk, "_mlite_dist_opt_sharded_state_dict", False):
            attach_checkpoint_metadata([chunk], ps)
        state.update({k: v for k, v in chunk.sharded_state_dict().items() if "lora" in k})
    return state


def save_lora_dcp(chunks, path, ps):
    from megatron.core import dist_checkpointing

    Path(path).mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save(_lora_sharded(chunks, ps), str(path))


def load_lora_dcp(chunks, path, ps):
    from megatron.core import dist_checkpointing

    dist_checkpointing.load(_lora_sharded(chunks, ps), str(path))


register_lora_adapter_type(GroupedLoRA)
register_lora_adapter_type(SharedOuterExpertLoRA)
