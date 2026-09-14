"""Exact synthetic-gradient TP optimizer scope probe. Run through Slurm.

Promoted from the validated scope probe; only assertion diagnostics change.
"""

import json, os, sys
from pathlib import Path
import torch
import torch.distributed as dist


def assert_optimizer_scope(current, norm, expected, step, local):
    """Compare the global norm first, then locate the first differing update."""
    from megatron.lite.model.qwen3_8_flash_next.tp import serial_parameter_name

    def first_parameter_mismatch():
        return next(
            (
                name
                for name, value in current.items()
                if not torch.equal(
                    value, local(name, expected['state'][serial_parameter_name(name)])
                )
            ),
            None,
        )

    details = {
        'step': step,
        'actual_norm_squared': float(norm) ** 2,
        'expected_norm_squared': float(expected['norm']) ** 2,
    }
    if float(norm) != expected['norm']:
        details['first_mismatched_parameter'] = first_parameter_mismatch()
        raise AssertionError(('TP_OPTIMIZER_LOGICAL_NORM', details))
    first = first_parameter_mismatch()
    assert first is None, (
        'TP_OPTIMIZER_LOGICAL_SCOPE',
        {**details, 'first_mismatched_parameter': first},
    )


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'unit' / 'model'))
    from test_qwen38_training import tiny_training_config
    from qwen38_dp_probe import reduced_gradients
    from megatron.lite.model.qwen3_8_flash_next.tp import (
        projection_shard,
        serial_parameter_name,
    )
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig

    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(1234)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    refpath = Path(sys.argv[2])
    cfgdir = out / f'cfg-{rank}'
    cfgdir.mkdir(exist_ok=True)
    (cfgdir / 'config.json').write_text(json.dumps(tiny_training_config()))
    cfg = MegatronLiteConfig(
        model_name='qwen3_8_flash_next',
        hf_path=str(cfgdir),
        load_hf_weights=False,
        parallel=ParallelConfig(tp=world, etp=1),
        optimizer=OptimizerConfig(lr=0.003, adam_eps=0.1),
        impl_cfg={
            'ngram_primes': (
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
            )
        },
    )
    runtime = MegatronLiteRuntime(str(cfgdir), cfg)
    handle = runtime.build_model()
    model = unwrap_model(handle._model)
    reference = torch.load(refpath, weights_only=True) if world > 1 else None

    def state():
        return {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}

    def local(n, v):
        return (
            v.chunk(world, 0)[rank].contiguous()
            if world > 1 and projection_shard(n)
            else v
        )

    if reference:
        model.load_state_dict(
            {
                n: local(n, reference['initial'][serial_parameter_name(n)])
                for n in model.state_dict()
            }
        )
        handle._optimizer.reload_model_params()
    initial = state()
    records = []
    numel = sum(p.numel() for p in model.parameters())
    if reference:
        assert numel < reference['numel'], ('TP_STORAGE', numel, reference['numel'])
    print(
        'TP_STORAGE',
        rank,
        numel,
        reference['numel'] if reference else numel,
        flush=True,
    )
    for step in range(2):
        runtime.zero_grad(handle)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            key = serial_parameter_name(n)
            value = {
                'lm_head.weight': 1 / 64,
                'embed_tokens.weight': 1 / 32,
                'layers.0.mlp.experts.fc1.weight0': 1 / 128,
            }.get(key, 0.0)
            p.main_grad.fill_(value * (step + 1))
        handle._extras['finalize_grads']()
        grads = reduced_gradients(handle._model, model)
        success, norm, _ = runtime.optimizer_step(handle)
        assert success
        current = state()
        record = dict(state=current, gradients=grads, norm=float(norm))
        records.append(record)
        torch.save(record, out / f'step{step}-rank{rank}.pt')
        if reference:
            expected = reference['steps'][step]
            assert_optimizer_scope(current, norm, expected, step, local)
            for n, v in grads.items():
                assert torch.equal(
                    v, local(n, expected['gradients'][serial_parameter_name(n)])
                ), ('TP_EXPERT_TOKEN_AVERAGE', n)
    runtime.save_checkpoint(handle, str(out / 'checkpoint'), step=2, save_rng=False)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    runtime.load_checkpoint(handle, str(out / 'checkpoint'), load_rng=False)
    for n, v in state().items():
        assert torch.equal(v, records[-1]['state'][n]), ('TP_CHECKPOINT_OWNER', n)
    if world == 1:
        assert not refpath.exists()
        torch.save(dict(initial=initial, steps=records, numel=numel), refpath)
    print('TP_OPTIMIZER_SCOPE_OK', world, rank, flush=True)
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
