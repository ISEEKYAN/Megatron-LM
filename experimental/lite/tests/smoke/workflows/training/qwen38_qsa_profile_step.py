"""真实Qwen runtime optimizer step归因；原policy不变，恢复同一步核prof未改数值。"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

sys.path.insert(
    0, os.environ['ARM_SRC'] + '/experimental/lite/tests/smoke/workflows/training'
)
from unittest.mock import patch

from megatron.lite.model.qwen3_8_flash_next import qsa
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.primitive.modules import gated_delta_net
from megatron.lite.primitive.ops.fla_l2norm import kernel_policy
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts import OptimizerConfig, PackedBatch
from qwen38_dp_probe import reduced_gradients, tiny_training_config
from qwen38_qsa_profile_observer import observed_attention


@record
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    assert int(os.environ['WORLD_SIZE']) == 1
    assert gated_delta_net._HAS_FLA, 'PROFILE_REAL_FLA_REQUIRED'
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    config = tiny_training_config()
    cfgdir = a.output / 'config'
    cfgdir.mkdir()
    (cfgdir / 'config.json').write_text(json.dumps(config))
    cfg = MegatronLiteConfig(
        model_name='qwen3_8_flash_next',
        hf_path=str(cfgdir),
        load_hf_weights=False,
        optimizer=OptimizerConfig(lr=0.003),
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
    observer = patch.object(
        qsa, 'sparse_attention', observed_attention(qsa.sparse_attention)
    )
    observer.start()
    runtime = MegatronLiteRuntime(str(cfgdir), cfg)
    handle = runtime.build_model()
    model = unwrap_model(handle._model)
    assert handle._parallel_state.cp_size == 1 and handle._parallel_state.tp_size == 1
    ids = [
        torch.arange(64, device='cuda') % 32,
        (torch.arange(64, device='cuda') * 3 + 47) % 127,
    ]
    batches = [
        PackedBatch(x, x.clone(), torch.tensor([64], device='cuda')) for x in ids
    ]

    def state():
        return {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}

    def snapshot(result):
        return dict(
            result=result,
            state=state(),
            gradients=reduced_gradients(handle._model, model),
        )

    def step():
        losses = []

        def loss_fn(output, batch):
            losses.append(output['loss'].detach())
            return output['loss'], {}

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        with torch.profiler.record_function('QWEN38_PROFILE_TRAIN_STEP'):
            runtime.zero_grad(handle)
            runtime.forward_backward(handle, batches, loss_fn, num_microbatches=2)
            success, norm, _ = runtime.optimizer_step(handle)
            end.record()
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - begin) * 1000
        event_ms = start.elapsed_time(end)
        assert success and 0 < float(norm) < float('inf'), 'PROFILE_OPTIMIZER_STEP'
        return dict(
            losses=[float(v) for v in losses],
            norm=float(norm),
            wall_ms=wall_ms,
            event_ms=event_ms,
        )

    raw = {}
    report = dict(
        config=config,
        policy=kernel_policy(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        microbatches=2,
        tokens_per_microbatch=64,
        warmup=[],
    )
    try:
        for i in range(4):
            result = step()
            report['warmup'].append(result)
            print('PROFILE_WARMUP_STEP', i, result, flush=True)
        runtime.save_checkpoint(
            handle, str(a.output / 'checkpoint'), step=4, save_rng=True
        )
        raw['before'] = state()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            result = step()
        prof.export_chrome_trace(str(a.output / 'trace.json'))
        raw['profiled'] = snapshot(result)
        assert (
            runtime.load_checkpoint(handle, str(a.output / 'checkpoint'), load_rng=True)
            == 4
        )
        for k, v in state().items():
            assert torch.equal(v, raw['before'][k]), ('PROFILE_CHECKPOINT_RESTORE', k)
        plain = step()
        raw['unprofiled'] = snapshot(plain)
        checks = 0
        for field in ['state', 'gradients']:
            for k, v in raw['profiled'][field].items():
                assert torch.equal(
                    v.contiguous().view(torch.uint8),
                    raw['unprofiled'][field][k].contiguous().view(torch.uint8),
                ), ('PROFILE_STEP_UNPERTURBED_BITWISE', field, k)
                checks += 1
        for k in ['losses', 'norm']:
            assert result[k] == plain[k], ('PROFILE_STEP_UNPERTURBED_SCALAR', k)
        trace = json.loads((a.output / 'trace.json').read_text())['traceEvents']
        kernels = [e for e in trace if e.get('cat') == 'kernel' and e.get('ph') == 'X']
        l2 = [
            e
            for e in kernels
            if 'l2norm_fwd_kernel' in e['name'] or 'l2norm_bwd_kernel' in e['name']
        ]
        fwd = [e for e in l2 if 'l2norm_fwd_kernel' in e['name']]
        bwd = [e for e in l2 if 'l2norm_bwd_kernel' in e['name']]
        assert len(fwd) == len(bwd) == 2, (
            'PROFILE_L2_REAL_KERNEL_COUNTS',
            len(fwd),
            len(bwd),
        )
        assert any(
            'chunk' in e['name'] for e in kernels
        ), 'PROFILE_FLA_RECURRENCE_KERNEL_REQUIRED'

        def union_us(events):
            intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in events)
            total = 0
            lo = hi = None
            for x, y in intervals:
                if lo is None:
                    lo, hi = x, y
                elif x <= hi:
                    hi = max(hi, y)
                else:
                    total += hi - lo
                    lo, hi = x, y
            return total + (hi - lo if lo is not None else 0)

        l2_us = sum(e['dur'] for e in l2)
        busy = union_us(kernels)
        markers = [
            e
            for e in trace
            if e.get('name') == 'QSA_KV_INDEX_BACKWARD'
            and e.get('ph') == 'X'
            and e.get('cat') == 'user_annotation'
        ]
        assert len(markers) == 4, ('QSA_INDEX_PROFILE_NATIVE_COUNTS', len(markers))
        gpu_markers = [
            e
            for e in trace
            if e.get('name') == 'QSA_KV_INDEX_BACKWARD'
            and e.get('cat') == 'gpu_user_annotation'
        ]
        assert len(gpu_markers) == 4 and {
            e['args']['External id'] for e in gpu_markers
        } == {
            e['args']['External id'] for e in markers
        }, 'QSA_INDEX_PROFILE_CPU_GPU_RANGE_IDENTITY'
        attributed = []
        ids = set()
        for marker in markers:
            children = [
                e
                for e in trace
                if e.get('cat') == 'cpu_op'
                and e.get('ph') == 'X'
                and e.get('tid') == marker['tid']
                and marker['ts'] <= e['ts']
                and e['ts'] + e['dur'] <= marker['ts'] + marker['dur'] + 0.01
            ]
            ids.update(
                e['args']['External id']
                for e in children
                if 'External id' in e.get('args', {})
            )
        attributed = [e for e in kernels if e.get('args', {}).get('External id') in ids]
        assert attributed and any(
            'indexing_backward' in e['name'] for e in attributed
        ), 'QSA_INDEX_PROFILE_KERNEL_CORRELATION'
        qsa_us = union_us(attributed)
        report.update(
            qsa_index_events=attributed,
            qsa_index_markers=markers,
            qsa_index_total_us=qsa_us,
            qsa_index_pct_unprofiled_step=100 * qsa_us / (plain['event_ms'] * 1000),
            qsa_index_pct_profiled_step=100 * qsa_us / (result['event_ms'] * 1000),
            qsa_index_pct_kernel_busy=100 * qsa_us / busy,
        )
        print(
            'QSA_REAL_STEP_INDEX_SHARE',
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k.startswith('qsa_index_')
                    and k not in ['qsa_index_events', 'qsa_index_markers']
                }
            ),
            flush=True,
        )

        report.update(
            profiled=result,
            unprofiled=plain,
            profile_unperturbed_tensor_checks=checks,
            l2_fwd_us=sum(e['dur'] for e in fwd),
            l2_bwd_us=sum(e['dur'] for e in bwd),
            l2_total_us=l2_us,
            kernel_busy_union_us=busy,
            l2_pct_profiled_step=100 * l2_us / (result['event_ms'] * 1000),
            l2_pct_unprofiled_step_denominator=100 * l2_us / (plain['event_ms'] * 1000),
            l2_pct_kernel_busy=100 * l2_us / busy,
            l2_events=l2,
            kernel_count=len(kernels),
        )
        print(
            'QWEN_REAL_TRAIN_STEP_L2_SHARE',
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k
                    not in [
                        'config',
                        'l2_events',
                        'warmup',
                        'qsa_index_events',
                        'qsa_index_markers',
                    ]
                }
            ),
            flush=True,
        )
        print('PROFILE_STEP_UNPERTURBED_BITWISE', checks, flush=True)
    finally:
        torch.save(raw, a.output / 'raw.pt')
        (a.output / 'report.json').write_text(json.dumps(report, indent=2))
        observer.stop()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
