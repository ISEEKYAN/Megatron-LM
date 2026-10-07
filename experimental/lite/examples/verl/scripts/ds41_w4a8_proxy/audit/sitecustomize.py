"""Evidence-only observers: retain upstream outputs and fail on a real strict-gate miss."""

import os

if os.environ.get('W4_GRPO_AUDIT') == '1':
    try:
        import hashlib
        import inspect
        import json
        import math
        import time
        import traceback
        from pathlib import Path

        import torch
        from verl_mlite.compat import apply_runtime_patches

        apply_runtime_patches()
        import verl.trainer.ppo.core_algos as algos
        import verl.utils.debug.metrics as debug
        from verl.utils.tracking import Tracking
        from verl_mlite.rollout.deepseek_v41 import ResyncReceiver

        root = Path(os.environ['W4_GRPO_OUT']) / 'audit'
        root.mkdir(parents=True, exist_ok=True)
        counters = {'raw': 0, 'adv': 0, 'resync': 0, 'optimizer': 0}

        def event(kind, record):
            record = dict(record, kind=kind, pid=os.getpid(), time=time.time())
            with (root / f'events-{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(record, default=str) + '\n')
            print('GRPO_AUDIT ' + json.dumps(record, default=str), flush=True)

        def resources():
            from pathlib import Path

            status = Path('/proc/self/status').read_text().splitlines()
            rss = int(next(x.split()[1] for x in status if x.startswith('VmRSS:')))
            return {
                'rss_kib': rss,
                'cuda_allocated': (
                    torch.cuda.memory_allocated()
                    if torch.cuda.is_initialized()
                    else None
                ),
                'cuda_peak_allocated': (
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_initialized()
                    else None
                ),
                'cuda_peak_reserved': (
                    torch.cuda.max_memory_reserved()
                    if torch.cuda.is_initialized()
                    else None
                ),
                'cuda_reserved': (
                    torch.cuda.memory_reserved()
                    if torch.cuda.is_initialized()
                    else None
                ),
            }

        def tensors(kind, record):
            target = root / f'{kind}-{os.getpid()}-{counters[kind]:03d}.pt'
            torch.save(record, target)
            counters[kind] += 1
            return str(target)

        raw_original = debug.calculate_debug_metrics

        def raw(data):
            batch = data.batch
            keys = ('old_log_probs', 'rollout_log_probs', 'response_mask', 'responses')
            assert all(k in batch for k in keys), list(batch.keys())
            saved = {
                k: v.detach().cpu().contiguous()
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            saved['non_tensor_batch'] = data.non_tensor_batch
            saved['meta_info'] = data.meta_info
            target = tensors('raw', saved)
            mask = saved['response_mask'][:, -saved['responses'].size(1) :].bool()
            left, right = saved['old_log_probs'][mask], saved['rollout_log_probs'][mask]
            assert (
                left.numel() > 0
                and torch.isfinite(left).all()
                and torch.isfinite(right).all()
            )
            diff = (left - right).abs()
            record = {
                'ordinal': counters['raw'],
                'tokens': left.numel(),
                'max_abs': diff.max().item(),
                'mismatch': (left != right).sum().item(),
                'tensor_file': target,
            }
            event('raw', record)
            assert torch.equal(
                left, right
            ), f'GRPO raw token logprob first difference: {record}'
            result = raw_original(data)
            assert result['training/rollout_probs_diff_max'] == 0, result
            return result

        debug.calculate_debug_metrics = raw
        adv_original = algos.compute_grpo_outcome_advantage
        adv_signature = inspect.signature(adv_original)

        def advantage(*args, **kwargs):
            bound = adv_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = adv_original(*args, **kwargs)
            rewards = bound.arguments['token_level_rewards'].detach().cpu()
            mask = bound.arguments['response_mask'].detach().cpu().bool()
            ids = bound.arguments['index']
            adv = result[0].detach().cpu()
            groups = {}
            for i, uid in enumerate(ids):
                groups.setdefault(str(uid), []).append(i)
            scores = rewards.sum(-1)
            nonzero = sum(
                bool((adv[rows][mask[rows]] != 0).any()) for rows in groups.values()
            )
            target = tensors(
                'adv',
                {
                    'token_level_rewards': rewards,
                    'response_mask': mask,
                    'advantages': adv,
                    'returns': result[1].detach().cpu(),
                    'index': ids,
                },
            )
            event(
                'adv',
                {
                    'ordinal': counters['adv'],
                    'groups': len(groups),
                    'nonzero_groups': nonzero,
                    'reward_min': scores.min().item(),
                    'reward_max': scores.max().item(),
                    'reward_std': scores.std().item(),
                    'rewards': scores.tolist(),
                    'tensor_file': target,
                },
            )
            assert (
                torch.isfinite(adv).all() and nonzero > 0
            ), 'GRPO has no nonzero-advantage groups'
            return result

        algos.compute_grpo_outcome_advantage = advantage
        algos.ADV_ESTIMATOR_REGISTRY['grpo'] = advantage
        import megatron.lite.primitive.optimizers.headwise_muon as headwise_muon

        MixedOptimizer = headwise_muon.MixedOptimizer

        optimizer_original = MixedOptimizer.step

        def master_digest(parameters):
            digest = hashlib.sha256()
            for param in parameters:
                assert param.dtype == torch.float32
                digest.update(param.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        def optimizer_step(self):
            params = [p for p in self.model.parameters() if p.requires_grad]
            before = master_digest(params)
            result = optimizer_original(self)
            after = master_digest(params)
            counters['optimizer'] += 1
            event(
                'optimizer',
                {
                    'ordinal': counters['optimizer'],
                    'success': bool(result[0]),
                    'grad_norm': float(result[1]),
                    'master_before': before,
                    'master_after': after,
                    'changed': before != after,
                    'trainable_parameters': len(params),
                    'pp_rank': self.ps.pp_rank,
                    'pp_size': self.ps.pp_size,
                    'ep_size': self.ps.ep_size,
                    'ep_rank': self.ps.ep_rank,
                    'dp_rank': self.ps.dp_rank,
                    'resources': resources(),
                    'optimizer_mode': (
                        'segmented_host'
                        if self.host_update is not None
                        else 'resident_transaction'
                    ),
                    'backend_types': [
                        type(backend).__name__ for backend in self.optimizers
                    ],
                    'host_update': (
                        self.host_update.last_metrics
                        if self.host_update is not None
                        else None
                    ),
                    'host_master_digest': (
                        master_digest([self.host_update.masters[id(p)] for p in params])
                        if self.host_update is not None
                        else None
                    ),
                    'optimizer_tensors_all_CPU': (
                        all(
                            value.device.type == 'cpu'
                            for backend in self.optimizers
                            for state in backend.state.values()
                            for value in state.values()
                            if isinstance(value, torch.Tensor)
                        )
                        if self.host_update is not None
                        else None
                    ),
                },
            )
            assert result[0] and math.isfinite(float(result[1])) and before != after
            if self.host_update is not None:
                assert (
                    master_digest([self.host_update.masters[id(p)] for p in params])
                    == after
                )
                assert all(
                    value.device.type == 'cpu'
                    for backend in self.optimizers
                    for state in backend.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                )
            return result

        MixedOptimizer.step = optimizer_step
        finish_original = ResyncReceiver.finish

        def finish(self):
            was_finished = self.finished
            result = finish_original(self)
            if not was_finished:
                counters['resync'] += 1
                event(
                    'resync',
                    {
                        'generation': counters['resync'],
                        'finished': self.finished,
                        'staging_current': self.staging.current_bytes,
                        'staging_peak': self.staging.peak_bytes,
                        'frozen_reuse': self.reuse_frozen,
                        'frozen_tables': len(self.expected_tables),
                        'received_tables': len(self.received_tables),
                        'resources': resources(),
                    },
                )
                assert self.finished and self.staging.current_bytes == 0
            return result

        ResyncReceiver.finish = finish
        log_original = Tracking.log

        def log(self, data, step, backend=None):
            run = getattr(self.logger.get('wandb'), 'run', None)
            url = getattr(run, 'url', None)
            if 'training/rollout_probs_diff_max' in data:
                event('step', {'step': step, 'metrics': data, 'wandb_url': url})
                assert data['training/rollout_probs_diff_max'] == 0
                assert (
                    'rollout_corr/k3_kl' in data and data['rollout_corr/k3_kl'] == 0
                ), data
                required = ['actor/loss', 'actor/grad_norm']
                for key in required:
                    assert key in data and math.isfinite(float(data[key])), (key, data)
                assert url and 'megatron-core-moe-dev' in url
            return log_original(self, data, step, backend)

        Tracking.log = log
        from verl_mlite.engine.mlite_engine import MegatronLiteEngine

        def state_hash(value):
            digest = hashlib.sha256()

            def visit(v):
                if isinstance(v, torch.Tensor):
                    digest.update(str((tuple(v.shape), v.dtype)).encode())
                    digest.update(
                        v.detach()
                        .cpu()
                        .contiguous()
                        .reshape(-1)
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
                elif isinstance(v, dict):
                    for key in sorted(v, key=str):
                        digest.update(str(key).encode())
                        visit(v[key])
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        visit(item)
                else:
                    digest.update(repr(v).encode())

            visit(value)
            return digest.hexdigest()

        def checkpoint_receipt(engine):
            import megatron.lite.primitive.ckpt.frozen_storage as frozen_storage

            storage_digest = frozen_storage.storage_digest

            model = engine.module
            opt = engine.handle._optimizer
            sched = engine.handle._lr_scheduler
            return {
                'rank': torch.distributed.get_rank(),
                'masters': master_digest(
                    [p for p in model.parameters() if p.requires_grad]
                ),
                'optimizer': state_hash(opt.state_dict()),
                'scheduler': state_hash(sched.state_dict()) if sched else None,
                'tables': {
                    name: storage_digest(m.weight, m.scale)
                    for name, m in model.named_modules()
                    if hasattr(m, 'master') and hasattr(m, 'lookup_fp8')
                },
            }

        save_original = MegatronLiteEngine.save_checkpoint
        load_original = MegatronLiteEngine.load_checkpoint

        def checkpoint_save(self, *args, **kwargs):
            before = checkpoint_receipt(self)
            result = save_original(self, *args, **kwargs)
            after = checkpoint_receipt(self)
            assert before == after, 'Checkpoint saving mutated live owners'
            event(
                'checkpoint_save',
                dict(after, path=str(args[0] if args else kwargs.get('local_path'))),
            )
            return result

        def checkpoint_load(self, *args, **kwargs):
            result = load_original(self, *args, **kwargs)
            event(
                'checkpoint_load',
                dict(
                    checkpoint_receipt(self),
                    path=str(args[0] if args else kwargs.get('local_path')),
                ),
            )
            return result

        MegatronLiteEngine.save_checkpoint = checkpoint_save
        MegatronLiteEngine.load_checkpoint = checkpoint_load
        event(
            'ready',
            {
                'training_sha': os.environ.get('W4_TRAIN_SHA'),
                'vllm_sha': os.environ.get('W4_VLLM_SHA'),
            },
        )
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(86)
