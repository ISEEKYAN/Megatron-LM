# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU transport lifecycle controls; no serving or numerical parity claim."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples/verl'))
from verl_mlite.rollout import ds41_lora as sync

pytestmark = pytest.mark.optional
BASE = 'a' * 64
CFG = dict(rank=2, alpha=4, layers=(0, 1), target_modules=('wq_b',))


def payload(version=0):
    source = {
        'layers.0.attn.wq_b.lora_A.weight': torch.ones(2, 4, dtype=torch.bfloat16),
        'layers.0.attn.wq_b.lora_B.weight': torch.full((4, 2), 0.5 + version, dtype=torch.bfloat16),
    }
    weights, config = sync.prepare_adapter_sync(source.items(), CFG, 1, version, BASE)
    return source, dict(weights), config


class Receiver:
    def __init__(self, tensors, terminal=True, duplicate=False):
        self.tensors, self.terminal, self.duplicate = tensors, terminal, duplicate
        self.drained = False

    def receive_weights(self, on_bucket_received):
        items = list(self.tensors.items())
        for index, (name, value) in enumerate(items):
            buffer = value.clone()
            batch = [(name, buffer)] * (2 if self.duplicate else 1)
            on_bucket_received(batch, self.terminal and index == len(items) - 1)
            buffer.fill_(-100)  # the real IPC receiver reuses its bucket buffer
        self.drained = True


@pytest.fixture
def runner(monkeypatch):
    request = ModuleType('vllm.lora.request')
    request.LoRARequest = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, request.__name__, request)
    for name in (
        'verl',
        'verl.workers',
        'verl.workers.rollout',
        'verl.workers.rollout.vllm_rollout',
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    utils = ModuleType('verl.workers.rollout.vllm_rollout.utils')
    utils.VLLM_LORA_INT_ID = 123
    utils.VLLM_LORA_NAME, utils.VLLM_LORA_PATH = '123', 'simon_lora_path'
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    calls = []

    def load(request, tensors, config):
        assert all('.lora_' in name for name in tensors)
        calls.append(('load', request, {k: v.clone() for k, v in tensors.items()}, config))

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: calls.append(('synchronize',)))
    model = type('DeepseekV41CPUModel', (), {})()
    return SimpleNamespace(
        model=model,
        device='cpu',
        calls=calls,
        lora_manager=SimpleNamespace(add_adapter_tensors=load),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(mlite_base_manifest_sha256=BASE))
        ),
    )


def test_three_adapter_generations_snapshot_and_ack_after_native_load(runner):
    manifests, names = [], []
    for version in range(3):
        source, tensors, config = payload(version)
        original = {k: v.clone() for k, v in source.items()}
        receiver = Receiver(tensors)
        ack = sync.receive_adapter(receiver, runner, config)
        assert receiver.drained and ack['finished'] and ack['staging_current'] == 0
        assert ack['policy_version'] == version and ack['base_manifest_sha256'] == BASE
        assert [c[0] for c in runner.calls[-2:]] == ['load', 'synchronize']
        loaded = runner.calls[-2]
        assert loaded[3]['lora_alpha'] == 4  # raw B, loader folds alpha/r exactly once
        assert '_mlite_ds41' not in loaded[3]
        assert loaded[3]['target_modules'] == ['language_model.model.layers.0.attn.wq_b']
        for key in source:
            assert torch.equal(source[key], original[key])
            assert torch.equal(loaded[2][key], original[key])
        manifests.append(ack['manifest_sha256'])
        names.append(ack['lora_name'])
        assert ack['lora_int_id'] == loaded[1].lora_int_id == 123
        assert loaded[1].lora_path == 'simon_lora_path'
    assert len(set(manifests)) == 3 and names == ['123'] * 3


@pytest.mark.parametrize(
    'bad', ['hash', 'base', 'partial', 'duplicate', 'no_terminal', 'version', 'no_header']
)
def test_invalid_publication_drains_without_loading_or_ack(runner, bad):
    _, tensors, config = payload()
    if bad == 'hash':
        config['_mlite_ds41']['sha256'] = '0' * 64
    elif bad == 'base':
        runner.vllm_config.model_config.hf_config.mlite_base_manifest_sha256 = 'b' * 64
    elif bad == 'partial':
        tensors.pop(next(iter(tensors)))
    elif bad == 'version':
        runner._mlite_ds41_adapter_ack = dict(policy_version=0)
    elif bad == 'no_header':
        config.pop('_mlite_ds41')
    receiver = Receiver(tensors, terminal=bad != 'no_terminal', duplicate=bad == 'duplicate')
    with pytest.raises(ValueError):
        sync.receive_adapter(receiver, runner, config)
    assert receiver.drained and not runner.calls


def test_publisher_rejects_base_duplicate_and_undeclared_identity():
    source, _, _ = payload()
    for weights, base in (
        ([('layers.0.attn.wq_b.weight', torch.ones(4, 4))], BASE),
        (list(source.items()) * 2, BASE),
        (source.items(), ''),
    ):
        with pytest.raises(ValueError):
            sync.prepare_adapter_sync(weights, CFG, 1, 0, base)


def test_existing_bucket_extension_and_non_adapter_fallback(runner, monkeypatch):
    utils = sys.modules['verl.workers.rollout.vllm_rollout.utils']
    original_calls = []

    class Extension:
        def update_weights_from_ipc(self, **kwargs):
            original_calls.append(kwargs)
            return 'original'

    utils.vLLMColocateWorkerExtension = Extension
    module = ModuleType('verl.workers.rollout.vllm_rollout.bucketed_weight_transfer')
    _, tensors, config = payload()
    module.BucketedWeightReceiver = lambda **kwargs: Receiver(tensors)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert sync.install() and not sync.install()
    worker = Extension()
    worker.device, worker.model_runner = 'cpu', runner
    worker._get_zmq_handle = lambda: 'existing-transport'
    assert worker.update_weights_from_ipc() == 'original' and len(original_calls) == 1
    ack = worker.update_weights_from_ipc(peft_config=config, base_sync_done=True)
    assert ack['finished'] and runner.calls[-1][0] == 'synchronize'


def test_real_engine_method_selects_only_shared_lora_export():
    pytest.importorskip('verl')
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    source, _, _ = payload()
    calls = []

    def export(handle, **kwargs):
        calls.append(kwargs)
        return iter(source.items())

    engine = SimpleNamespace(
        _require_initialized=lambda: None,
        is_param_offload_enabled=False,
        _initial_sync_cache_cleared=True,
        _resolve_model_name=lambda: 'deepseek_v41',
        engine_config=SimpleNamespace(impl_cfg=dict(lora=CFG)),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(mlite_base_manifest_sha256=BASE)),
        handle=SimpleNamespace(_extras=dict(model_cfg=SimpleNamespace(num_hidden_layers=1))),
        runtime=SimpleNamespace(export_weights=export),
    )
    for version in range(3):
        weights, config = MegatronLiteEngine.get_per_tensor_param(engine, base_sync_done=True)
        assert set(dict(weights)) == set(source)
        assert config['_mlite_ds41']['manifest']['policy_version'] == version
    assert calls == [dict(target='lora')] * 3
    with pytest.raises(ValueError, match='frozen release base'):
        MegatronLiteEngine.get_per_tensor_param(engine, base_sync_done=False)
    assert len(calls) == 3  # never falls back to a full base publication
