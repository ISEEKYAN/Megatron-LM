"""Real Ray ObjectRefs and ZMQ cover RPC failures without changing wire order."""

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.optional


@pytest.fixture(scope='module')
def ray_runtime():
    ray = pytest.importorskip('ray')
    pytest.importorskip('zmq')
    os.environ['RAY_AUTH_MODE'] = 'disabled'
    ray.init(num_cpus=2, include_dashboard=False, _temp_dir='/tmp/rf')
    yield ray
    ray.shutdown()


@pytest.mark.parametrize('case', ['healthy', 'preconnect', 'ack_wait', 'no_ref'])
@pytest.mark.parametrize('prefetch', [False, True])
def test_original_rpc_error_does_not_strand_sender(
    ray_runtime, tmp_path, monkeypatch, case, prefetch
):
    import torch
    import zmq
    from verl_mlite import compat as C

    C._patch_transformers_vision2seq_alias()
    C._patch_transformers_rope_ignore_keys()
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as B
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    C._patch_sender_rpc_failure()
    assert ServerAdapter._execute_method._mlite_rpc_failure

    ray = ray_runtime
    handle = f'ipc://{tmp_path}/wire.sock'

    class Peer:
        def __init__(self):
            self.calls = 0

        def ready(self):
            import torch

            assert torch.bfloat16 is not None
            return case

        def call_count(self):
            return self.calls

        def collective_rpc(self, method, **kwargs):
            self.calls += 1
            if case == 'preconnect':
                raise ValueError('original-before-connect')
            ctx = zmq.Context()
            socket = ctx.socket(zmq.REP)
            socket.setsockopt(zmq.RCVTIMEO, 10000)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(handle)
            try:
                assert socket.recv_pyobj() == 'cpu-buffer'
                if case == 'ack_wait':
                    raise ValueError('original-before-ack')
                socket.send(b'ACK')
                buckets = []
                while True:
                    value = socket.recv_pyobj()
                    buckets.append(value)
                    socket.send(b'ACK')
                    if value['is_last']:
                        return buckets
            finally:
                socket.close()
                ctx.term()

    peer_class = ray.remote(Peer)
    peer = peer_class.options(
        name='failfast-' + __import__('uuid').uuid4().hex, lifetime='detached'
    ).remote()
    assert ray.get(peer.ready.remote()) == case
    original_execute = ServerAdapter._execute_method
    while getattr(original_execute, '_mlite_rpc_failure', False):
        original_execute = original_execute.__wrapped__

    class Adapter(ServerAdapter):
        _execute_method = original_execute

    class Sender(B.BucketedWeightSender):
        def _init_socket(self):
            super()._init_socket()
            underlying = self.socket
            underlying.setsockopt(zmq.SNDTIMEO, 5000)
            underlying.setsockopt(zmq.RCVTIMEO, 5000)

            class RecordingSocket:
                def __getattr__(self, name):
                    return getattr(underlying, name)

                def send_pyobj(self, *args, **kwargs):
                    wire.append(args[0])
                    return underlying.send_pyobj(*args, **kwargs)

            self.socket = RecordingSocket()

        def _init_buffer(self):
            self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8)
            self.socket.send_pyobj('cpu-buffer')
            self.socket.recv()

    wire = []
    Sender._mlite_weight_prefetch_patch = False
    if prefetch:
        C._install_bucketed_sender_prefetch(Sender)
    C._install_sender_rpc_failure(Adapter, Sender)
    monkeypatch.setattr(
        B,
        'get_torch_device',
        lambda: SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None),
    )
    monkeypatch.setattr(B, 'is_support_ipc', lambda: False)
    stopped_at = []
    original_call = C._RPCFailureSocket._call

    def observed_call(self, *args, **kwargs):
        try:
            return original_call(self, *args, **kwargs)
        except BaseException:
            stopped_at.append(time.monotonic())
            raise

    monkeypatch.setattr(C._RPCFailureSocket, '_call', observed_call)
    adapter = Adapter.__new__(Adapter)
    adapter._has_server = case != 'no_ref'
    adapter.server_handle = peer
    sender = Sender(handle, bucket_size_mb=1, use_shm=not prefetch)
    sender._mlite_prefetch_allow_cpu = True
    weights = [('layers.0.x', torch.arange(8, dtype=torch.bfloat16))]

    async def run():
        ref = await adapter._execute_method('update_weights_from_ipc', non_block=True)
        if ref is None:
            # Nonleaders have no local RPC; the real peer still connects normally.
            peer_ref = peer.collective_rpc.remote('update_weights_from_ipc')
        else:
            peer_ref = ref
        original_error = []
        completed_at = []

        def complete(future):
            completed_at.append(time.monotonic())
            try:
                future.result()
            except BaseException as error:
                original_error.append(error)

        peer_ref.future().add_done_callback(complete)
        started = time.monotonic()
        if case in ('healthy', 'no_ref'):
            await sender.async_send_weights(iter(weights))
            returned = ray.get(peer_ref)
            assert len(returned) == 1 and returned[0]['is_last'] is True
            assert list(returned[0]['bucket_meta']) == ['layers.0.x']
            assert wire[0] == 'cpu-buffer'
            assert wire[1:] == [
                {k: v for k, v in b.items() if k != '_w10_policy_receipt'} for b in returned
            ]
            if case == 'no_ref':
                assert C._SENDER_RPC_FAILURE.get() is None
        else:
            with pytest.raises(ray.exceptions.RayTaskError) as caught:
                await sender.async_send_weights(iter(weights))
            failure = C._SENDER_RPC_FAILURE.get()
            assert caught.value is failure[0]
            assert f'original-before-{"connect" if case == "preconnect" else "ack"}' in str(
                caught.value
            )
            # Detect at the wire boundary: native finally/GC runs afterwards.
            assert completed_at and stopped_at
            assert stopped_at[0] - completed_at[0] < 0.5
            assert not wire or wire == ['cpu-buffer']
        assert time.monotonic() - started < 15

    try:
        asyncio.run(run())
        assert ray.get(peer.call_count.remote()) == 1
    finally:
        ray.kill(peer)
