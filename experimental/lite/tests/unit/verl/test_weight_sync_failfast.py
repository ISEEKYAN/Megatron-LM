# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Peer errors must escape ACK waits; a timeout can never acknowledge weights."""
import asyncio
import importlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def compat(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / 'examples/verl'))
    return importlib.import_module('verl_mlite.compat')


def test_rpc_hook_is_paired_idempotent_and_preserves_return(compat):
    result = Future()

    class Adapter:
        async def _execute_method(self, method, **kwargs):
            return result

        async def update_weights(self):
            return await self._execute_method('update_weights_from_ipc', non_block=True)

    execute, update = Adapter._execute_method, Adapter.update_weights
    assert compat._install_rollout_weight_failure_watch(Adapter)
    installed = Adapter._execute_method
    assert not compat._install_rollout_weight_failure_watch(Adapter)
    assert Adapter._execute_method is installed
    assert asyncio.run(Adapter().update_weights()) is result
    assert compat._uninstall_rollout_weight_failure_watch(Adapter)
    assert Adapter._execute_method is execute and Adapter.update_weights is update
    assert not compat._uninstall_rollout_weight_failure_watch(Adapter)


def test_failed_actor_rejected_before_next_update(compat):
    calls, result = [], Future()
    error = RuntimeError('receiver failed to load bucket')

    class Adapter:
        async def _execute_method(self, method, **kwargs):
            return result

        async def update_weights(self):
            calls.append('update')
            await self._execute_method('update_weights_from_ipc', non_block=True)
            result.set_exception(error)
            compat._WEIGHT_SYNC_RPC.get().check()

    compat._install_rollout_weight_failure_watch(Adapter)
    actor = Adapter()
    with pytest.raises(RuntimeError) as first:
        asyncio.run(actor.update_weights())
    assert first.value is error and actor._mlite_weight_sync_failed is error
    with pytest.raises(RuntimeError, match='cannot be reused') as second:
        asyncio.run(actor.update_weights())
    assert second.value.__cause__ is error and calls == ['update']
    compat._uninstall_rollout_weight_failure_watch(Adapter)


def test_ray_shaped_future_reports_original_exception_from_other_thread(compat):
    completion = Future()
    abort = compat._WeightSyncAbort()
    abort.watch(SimpleNamespace(future=lambda: completion))
    error = MemoryError('injected peer allocation failure')
    thread = threading.Thread(target=lambda: completion.set_exception(error))
    thread.start()
    assert abort.event.wait(0.5)
    thread.join()
    with pytest.raises(MemoryError) as caught:
        abort.check()
    assert caught.value is error


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'bad'])
def test_invalid_single_ack_failure_timeout_rejected(compat, monkeypatch, value):
    monkeypatch.setenv('MLITE_WEIGHT_SYNC_SINGLE_ACK_TIMEOUT_SECONDS', value)
    with pytest.raises(ValueError):
        compat._AbortableWeightSocket(object(), compat._WeightSyncAbort())


def test_ack_timeout_is_failure_not_success(compat, monkeypatch):
    zmq = pytest.importorskip('zmq')
    monkeypatch.setenv('MLITE_WEIGHT_SYNC_SINGLE_ACK_TIMEOUT_SECONDS', '0.05')
    ctx = zmq.Context()
    req, rep = ctx.socket(zmq.REQ), ctx.socket(zmq.REP)
    req.bind('inproc://missing-ack')
    rep.connect('inproc://missing-ack')
    abort = compat._WeightSyncAbort()
    socket = compat._AbortableWeightSocket(req, abort)
    try:
        socket.send_pyobj({'bucket': 0})
        with pytest.raises(TimeoutError) as caught:
            socket.recv()
        assert abort.error is caught.value and abort.event.is_set()
    finally:
        req.close(linger=0)
        rep.close(linger=0)
        ctx.term()


def test_each_ack_gets_new_deadline_even_when_total_exceeds_it(compat, monkeypatch):
    zmq = pytest.importorskip('zmq')
    monkeypatch.setenv('MLITE_WEIGHT_SYNC_SINGLE_ACK_TIMEOUT_SECONDS', '0.2')
    ctx = zmq.Context()
    req = ctx.socket(zmq.REQ)
    req.bind('inproc://separate-ack-deadlines')

    def receive():
        rep = ctx.socket(zmq.REP)
        rep.connect('inproc://separate-ack-deadlines')
        try:
            for index in range(3):
                assert rep.recv_pyobj() == {'bucket': index}
                time.sleep(0.08)
                rep.send(b'ack')
        finally:
            rep.close(linger=0)

    thread = threading.Thread(target=receive)
    thread.start()
    socket = compat._AbortableWeightSocket(req, compat._WeightSyncAbort())
    start = time.monotonic()
    try:
        for index in range(3):
            socket.send_pyobj({'bucket': index})
            assert socket.recv() == b'ack'
        assert time.monotonic() - start > 0.2
    finally:
        req.close(linger=0)
        thread.join(timeout=1)
        assert not thread.is_alive()
        ctx.term()


def test_cleanup_retains_borrowed_storage_until_worker_stops(compat, monkeypatch):
    monkeypatch.setattr(compat, '_WEIGHT_SYNC_SHUTDOWN_TIMEOUT_S', 0.05)
    release = threading.Event()
    calls = []
    executor = ThreadPoolExecutor(max_workers=1)
    worker = executor.submit(release.wait)
    sender = SimpleNamespace(
        socket=SimpleNamespace(close=lambda **kwargs: calls.append('socket')),
        _cleanup=lambda: calls.append('cleanup'),
    )
    staging = [object(), object()]
    start = time.monotonic()
    compat._finish_weight_sender(
        sender, executor, worker, staging, compat._WeightSyncAbort()
    )
    assert time.monotonic() - start < 0.5
    assert calls == ['socket'] and id(worker) in compat._DEFERRED_WEIGHT_SENDERS
    assert compat._DEFERRED_WEIGHT_SENDERS[id(worker)][2] is staging
    release.set()
    worker.result(timeout=1)
    for _ in range(100):
        if 'cleanup' in calls:
            break
        time.sleep(0.001)
    assert (
        calls == ['socket', 'cleanup']
        and id(worker) not in compat._DEFERRED_WEIGHT_SENDERS
    )


def test_eager_runtime_installer_watches_already_imported_rollout(compat, monkeypatch):
    import sys

    class Adapter:
        async def _execute_method(self, method, **kwargs):
            return None

        async def update_weights(self):
            return None

    module = SimpleNamespace(ServerAdapter=Adapter)
    monkeypatch.setitem(sys.modules, compat._VLLM_ROLLOUT_MODULE, module)
    monkeypatch.delitem(sys.modules, compat._BUCKETED_SENDER_MODULE, raising=False)
    monkeypatch.setattr(sys, 'meta_path', list(sys.meta_path))
    assert compat._patch_bucketed_weight_transfer()
    assert not compat._patch_bucketed_weight_transfer()
    assert compat._uninstall_rollout_weight_failure_watch(Adapter)


def test_lazy_loader_watches_rollout_instead_of_its_sender_alias(compat):
    class Adapter:
        async def _execute_method(self, method, **kwargs):
            return None

        async def update_weights(self):
            return None

    original = Adapter._execute_method
    module = SimpleNamespace(
        __name__=compat._VLLM_ROLLOUT_MODULE,
        ServerAdapter=Adapter,
        BucketedWeightSender=object(),
    )
    loader = SimpleNamespace(exec_module=lambda module: None)
    compat._SenderPatchLoader(loader).exec_module(module)
    assert Adapter._execute_method is not original
    assert compat._uninstall_rollout_weight_failure_watch(Adapter)
    assert Adapter._execute_method is original
