"""Minimal executable usage of the dense multi-LoRA bank."""

import torch
from megatron.lite.primitive.modules.multi_lora_bank import DenseLoraBank


def main() -> None:
    torch.manual_seed(0)
    a_bank = torch.randn(2, 2, 3)
    b_bank = torch.randn(2, 4, 2)
    tokens = torch.randn(5, 3)
    adapter_for_token = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)

    delta = DenseLoraBank(a_bank, b_bank).delta(tokens, adapter_for_token, scale=0.5)
    assert delta.shape == (5, 4)
    print(delta.shape)


if __name__ == "__main__":
    main()
