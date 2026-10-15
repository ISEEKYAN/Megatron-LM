# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Worker wiring forwards every IPC bucket and finalizes once per generation."""
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from test_receiver_staging import adapter


@pytest.mark.parametrize(
    'peft,drafter', [(None, None), (object(), None), (None, object())]
)
def test_worker_extension_ipc_generation(v41_core_te, monkeypatch, peft, drafter):
    consumer = adapter(v41_core_te)
    events = []
    monkeypatch.setattr(
        consumer, 'install_reload_metadata_hook', lambda: events.append('metadata')
    )
    model, config = object(), object()
    buckets = [[('weight', torch.ones(2))], [('scale', torch.ones(1))]]

    class Receiver:
        def __init__(self, actual_model, actual_config):
            assert actual_model is model and actual_config is config
            events.append('receiver')

        def receive(self, weights):
            events.append(weights)

        def finish(self):
            events.append('finish')

    class Transport:
        def __init__(self, **kwargs):
            assert kwargs == dict(zmq_handle='handle', device='cpu', use_shm=True)

        def receive_weights(self, on_bucket_received):
            for i, weights in enumerate(buckets):
                on_bucket_received(weights, i == len(buckets) - 1)

    monkeypatch.setattr(consumer, 'ResyncReceiver', Receiver)
    parent = None
    for name in (
        'verl',
        'verl.workers',
        'verl.workers.rollout',
        'verl.workers.rollout.vllm_rollout',
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
        if parent is not None:
            setattr(parent, name.rsplit('.', 1)[-1], package)
        parent = package
    utils = ModuleType(parent.__name__ + '.utils')
    utils.vLLMColocateWorkerExtension = object
    transfer = ModuleType(parent.__name__ + '.bucketed_weight_transfer')
    transfer.BucketedWeightReceiver = Transport
    for module in (utils, transfer):
        monkeypatch.setitem(sys.modules, module.__name__, module)
        setattr(parent, module.__name__.rsplit('.', 1)[-1], module)
    worker = consumer.worker_extension()()
    worker.model_runner = NS(
        model=model, vllm_config=NS(model_config=config, speculative_config=drafter)
    )
    worker.device = 'cpu'
    worker._get_zmq_handle = lambda: 'handle'
    assert events == ['metadata']
    if peft or drafter:
        with pytest.raises(NotImplementedError, match='without a drafter'):
            worker.update_weights_from_ipc(peft_config=peft, use_shm=True)
        assert events == ['metadata']
    else:
        worker.update_weights_from_ipc(use_shm=True)
        assert events == ['metadata', 'receiver', *buckets, 'finish']
