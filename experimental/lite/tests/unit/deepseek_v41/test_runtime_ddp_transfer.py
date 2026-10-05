# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native model-owned DDP survives the runtime's parameter residency boundary."""
import runpy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, directory, cuda):
    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig
    from megatron.lite.runtime import megatron_utils
    from megatron.lite.runtime.backends.mlite import runtime as runtime_owner
    from megatron.lite.runtime.contracts import ParallelConfig
    from test_redo_parity import release_config

    torch.set_num_threads(1)
    if cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl' if cuda else 'gloo',
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    cfg = release_config().to_hf_dict()
    cfg['text_config'].update(
        num_hidden_layers=2,
        compress_ratios=[0] * 5,
        candidate_source_layer_id=-1,
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        engram_layer_ids=[1],
        engram_num_embeddings=[18],
    )
    bundle = protocol.build_model(
        DeepseekV41Config(cfg),
        impl_cfg=protocol.ImplConfig(
            device='cuda' if cuda else 'cpu',
            dtype=torch.float32,
            quantized=False,
            optimizer='muon',
            optimizer_config=OptimizerConfig(
                lr=1e-6, ns_steps=2, coefficient_type='quintic'
            ),
            parallel=ParallelConfig(ep=2),
            token_map=list(range(64)),
        ),
    )
    model = bundle.chunks[0]
    # A controlled owner-only loss isolates synchronization from model math.
    # All ranks differentiate every embedding element; the independent global
    # mean has derivative (1 + 2)/2 = 1.5, exactly representable in FP32.
    model.forward = lambda x: model.embed.weight.sum() * x
    identities = {name: id(p) for name, p in model.named_parameters()}
    owners = {id(p) for group in bundle.optimizer.param_groups for p in group['params']}
    handle = SimpleNamespace(
        _model=model,
        _optimizer=bundle.optimizer,
        _extras={**bundle.extras, 'model_chunks': [model]},
    )
    runtime = runtime_owner.MegatronLiteRuntime.__new__(
        runtime_owner.MegatronLiteRuntime
    )

    def check_gradient():
        bundle.optimizer.zero_grad(set_to_none=True)
        execution = bundle.forward_step.keywords['execution_model']
        execution(
            torch.tensor(float(rank + 1), device=model.embed.weight.device)
        ).backward()
        assert torch.equal(
            model.embed.weight.grad, torch.full_like(model.embed.weight, 1.5)
        )
        assert identities == {name: id(p) for name, p in model.named_parameters()}
        assert owners == {
            id(p) for group in bundle.optimizer.param_groups for p in group['params']
        }

    check_gradient()
    if cuda:
        for _ in range(2):
            bundle.optimizer.zero_grad(set_to_none=True)
            runtime.to(handle, 'cpu', optimizer=False, grad=False)
            assert model.embed.weight.device.type == 'cpu'
            runtime.to(handle, 'cuda', optimizer=False, grad=False)
            check_gradient()
    else:
        # Zero-GPU dtype migration replaces AccumulateGrad, like device migration;
        # it does not claim a CUDA offload numerical result.
        with patch.object(
            megatron_utils,
            'offload_model_to_cpu',
            lambda chunks: [chunk.double() for chunk in chunks],
        ), patch.object(
            megatron_utils,
            'load_model_to_gpu',
            lambda chunks, **kwargs: [chunk.float() for chunk in chunks],
        ):
            for _ in range(2):
                bundle.optimizer.zero_grad(set_to_none=True)
                runtime.to(handle, 'cpu', optimizer=False, grad=False)
                runtime.to(handle, 'cuda', optimizer=False, grad=False)
                check_gradient()
    dist.destroy_process_group()


def test_runtime_ddp_rebind_after_cpu_dtype_migration(v41_core_te, tmp_path):
    mp.start_processes(
        _worker, args=(str(tmp_path), False), nprocs=2, join=True, start_method='spawn'
    )


@pytest.mark.gpus(2)
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason='two CUDA devices for residency'
)
def test_runtime_ddp_rebind_after_actual_cuda_offload(v41_core_te, tmp_path):
    mp.start_processes(
        _worker, args=(str(tmp_path), True), nprocs=2, join=True, start_method='spawn'
    )
