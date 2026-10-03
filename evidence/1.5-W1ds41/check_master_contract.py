"""CPU-only blocker probe; no model, CUDA, or optimizer backend initialization."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
PRIMITIVE = ROOT / "experimental/lite/megatron/lite/primitive"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codec = load("master_probe_mxfp4", PRIMITIVE / "quantization/mxfp4.py")
linear = load("master_probe_linear", PRIMITIVE / "modules/native_fp32_linear.py")
# Execute the production validation method unchanged, without importing the
# optional emerging-optimizers backend. This is not an optimizer-step test.
source = PRIMITIVE / "optimizers/headwise_muon.py"
tree = ast.parse(source.read_text())
cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HeadwiseMuon"
)
method = next(
    n
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == "_validate_groups"
)
shape = next(
    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_matrix_shape"
)
namespace = {"torch": torch, "math": __import__("math")}
exec(
    compile(ast.Module(body=[shape, method], type_ignores=[]), str(source), "exec"),
    namespace,
)
results = {}
for dtype in (torch.float32, torch.bfloat16):
    p = torch.nn.Parameter(torch.ones(2, 32, dtype=dtype))
    owner = SimpleNamespace(
        param_groups=[
            dict(
                params=[p],
                matrix_shape=(2, 32),
                lr=0.01,
                weight_decay=0.1,
                momentum=0.95,
                update_rms=0.18,
            )
        ]
    )
    try:
        namespace["_validate_groups"](owner)
        results[str(dtype)] = "accepted"
    except ValueError as error:
        results[str(dtype)] = str(error)
assert results["torch.float32"] == "accepted"
assert "FP32 master" in results["torch.bfloat16"]

# Same deploy codec, different quantization inputs. BF16 rounding moves the
# first value onto a ties-down boundary while the block scale stays exactly 1.
w = torch.zeros(1, 32, dtype=torch.float32)
w[0, 0], w[0, 1] = 0.7501, 6.0
packed32, scale32 = codec.quantize_mxfp4(w)
packed16, scale16 = codec.quantize_mxfp4(w.bfloat16())
assert packed32.view(torch.uint8)[0, 0].item() == 114
assert packed16.view(torch.uint8)[0, 0].item() == 113
assert torch.equal(scale32.view(torch.uint8), scale16.view(torch.uint8))

# Native FP32 provider really publishes FP32 gradients with BF16 residuals.
p = torch.nn.Parameter(torch.ones(2, 32, dtype=torch.float32))
x = torch.ones(3, 32, dtype=torch.bfloat16, requires_grad=True)
linear.native_fp32_linear(x, p).sum().backward()
assert p.grad.dtype == torch.float32 and x.grad.dtype == torch.bfloat16
try:
    linear.native_fp32_linear(x, p.bfloat16())
except ValueError as error:
    rejection = str(error)
else:
    raise AssertionError("BF16 master unexpectedly accepted")

print(
    json.dumps(
        dict(
            torch=torch.__version__,
            device="cpu",
            validation=results,
            fp32_packed_byte=packed32.view(torch.uint8)[0, 0].item(),
            bf16_packed_byte=packed16.view(torch.uint8)[0, 0].item(),
            scale_byte=scale32.view(torch.uint8)[0, 0].item(),
            native_weight_gradient_dtype=str(p.grad.dtype),
            native_bf16_rejection=rejection,
        ),
        indent=2,
    )
)
