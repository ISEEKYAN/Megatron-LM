"""Native Nemotron-H layer composition; runtime owns scheduling and collectives."""

import torch
from torch import nn

from .attention import Attention
from .experts import MoE
from .functional import RMSNorm, linear
from .mamba import MambaMixer, SSMMeta


class Block(nn.Module):
    def __init__(self, config, ps, layer, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.norm = RMSNorm(
            config.hidden_size, config.layer_norm_epsilon, device=device, dtype=dtype
        )
        self.kind = config.layers_block_type[layer]
        types = {
            "linear_attention": MambaMixer,
            "full_attention": Attention,
            "moe": MoE,
        }
        if self.kind not in types:
            raise ValueError(f"Unsupported Nemotron block: {self.kind}")
        self.mixer = types[self.kind](config, ps, device=device, dtype=dtype)

    def forward(self, hidden, residual, meta):
        if residual is None:
            residual, hidden = hidden, self.norm(hidden)
        else:
            hidden, residual = self.norm(hidden, residual)
        hidden = self.mixer(hidden) if self.kind == "moe" else self.mixer(hidden, meta)
        return hidden, residual


class NemotronModel(nn.Module):
    """A contiguous pipeline stage with explicit hidden/residual boundary state.

    Intermediate stages return both streams as [tokens, 1, 2*hidden]. No broadcast
    or pipeline scheduler is implemented here; the mlite runtime must transport
    this payload and supply it through set_input_tensor.
    """

    def __init__(
        self, config, ps, *, layer_range=None, device=None, dtype=torch.bfloat16
    ):
        super().__init__()
        self.config, self.ps = config, ps
        start, end = layer_range or (0, config.num_hidden_layers)
        if not 0 <= start < end <= config.num_hidden_layers:
            raise ValueError("Invalid contiguous layer range")
        if ps.pp_size > 1 and layer_range is None:
            raise ValueError("PP requires an explicit layer assignment from runtime")
        self.pre_process = start == 0
        self.post_process = end == config.num_hidden_layers
        if config.tie_word_embeddings:
            raise ValueError("Tied embedding synchronization is not implemented")
        self.share_embeddings_and_output_weights = False
        self.embeddings = (
            nn.Embedding(
                config.vocab_size, config.hidden_size, device=device, dtype=dtype
            )
            if self.pre_process
            else None
        )
        self.layers = nn.ModuleDict(
            {
                str(i): Block(config, ps, i, device=device, dtype=dtype)
                for i in range(start, end)
            }
        )
        self.norm_f = (
            RMSNorm(
                config.hidden_size,
                config.layer_norm_epsilon,
                device=device,
                dtype=dtype,
            )
            if self.post_process
            else None
        )
        self.lm_head = (
            nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
            if self.post_process
            else None
        )
        self._input_tensor = None

    def set_input_tensor(self, tensor):
        if isinstance(tensor, list):
            if len(tensor) != 1:
                raise ValueError("Nemotron stage expects one packed pipeline payload")
            tensor = tensor[0]
        self._input_tensor = tensor

    def forward(self, input_ids, *, meta: SSMMeta):
        if self.pre_process:
            hidden, residual = self.embeddings(input_ids.reshape(-1)), None
        else:
            if self._input_tensor is None:
                raise ValueError("Missing pipeline hidden/residual payload")
            if self._input_tensor.shape[1:] != (1, 2 * self.config.hidden_size):
                raise ValueError("Expected [tokens, 1, 2*hidden] pipeline payload")
            hidden, residual = self._input_tensor[:, 0].chunk(2, dim=-1)
            self._input_tensor = None
        for layer in self.layers.values():
            hidden, residual = layer(hidden, residual, meta)
        if not self.post_process:
            return torch.cat((hidden, residual), dim=-1).unsqueeze(1)
        hidden, _ = self.norm_f(hidden, residual)
        return linear(hidden, self.lm_head.weight)
