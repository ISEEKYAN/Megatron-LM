# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch


@pytest.mark.parametrize(
    'quantized,w4a8', [(True, False), (False, True), (False, False)]
)
def test_deployment_math_rejects_unverified_expert_modes(v41_core_te, quantized, w4a8):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model
    from test_w4a8_fp32 import tiny_config

    config = tiny_config()
    with pytest.raises(ValueError, match='Deployment math requires quantized W4A8'):
        protocol.build_model(
            config,
            impl_cfg=protocol.ImplConfig(
                device='cpu',
                dtype=torch.bfloat16,
                quantized=quantized,
                w4a8_experts=w4a8,
                deployment_math=True,
            ),
        )
    with pytest.raises(ValueError, match='Deployment math requires quantized W4A8'):
        DeepseekV41Model(
            config, quantized=quantized, w4a8_experts=w4a8, deployment_math=True
        )


def test_deployment_math_rejects_fp32_residuals(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from test_w4a8_fp32 import tiny_config

    with pytest.raises(ValueError, match='Deployment math requires'):
        protocol.build_model(
            tiny_config(),
            impl_cfg=protocol.ImplConfig(
                device='cpu', quantized=False, deployment_math=True, dtype=torch.float32
            ),
        )


def test_documented_w4a8_recipe_constructs(v41_core_te, tmp_path, monkeypatch):
    from pathlib import Path

    from test_w4a8_fp32 import tiny_config

    path = Path(__file__).resolve().parents[3] / 'examples/verl/DS41_W4A8_VALIDATION.md'
    # Substitute CPU/EP1 only; keep quantized, dtype and optimizer settings.
    import ast
    import json

    source = path.read_text().split('```python\n')[1].split('```')[0]
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for keyword in node.keywords:
                if node.func.id == 'ImplConfig' and keyword.arg == 'device':
                    keyword.value = ast.Constant('cpu')
                if node.func.id == 'ParallelConfig' and keyword.arg == 'ep':
                    keyword.value = ast.Constant(1)
    (tmp_path / 'config.json').write_text(json.dumps(tiny_config().to_hf_dict()))
    monkeypatch.chdir(tmp_path)
    scope = {}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), scope)
    assert scope['impl'].optimizer_config.lr == 1e-6
    model = scope['bundle'].chunks[0]
    assert model.deployment_math
    assert scope['bundle'].optimizer is not None
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
