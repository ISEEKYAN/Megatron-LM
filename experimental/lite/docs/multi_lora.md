# Multi-LoRA

```python
import torch
from megatron.lite.primitive.modules.multi_lora_bank import DenseLoraBank

a = torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2, 32, 4, device="cuda", dtype=torch.bfloat16)
x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
indices = torch.zeros(8, device="cuda", dtype=torch.int64)

bank = DenseLoraBank(a, b)
delta = bank.delta(x, indices, scale=0.25)
```
