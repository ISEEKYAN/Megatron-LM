# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
import threading

import pytest
import torch

from verl_mlite.compat import (
    _BUCKETED_SENDER_MODULE,
    _install_bucketed_sender_prefetch,
    _instrument_bucketed_weight_sender,
    _patch_bucketed_weight_sender,
    _weight_sync_probe_enabled,
)


class _Socket:
    def send_pyobj(self, value):
        return value

    def recv(self):
        return b"ok"


class _Sender:
    def _init_socket(self):
        self.socket = _Socket()

    async def async_send_weights(self, weights):
        self._init_socket()
        self.socket.send_pyobj({"is_last": True})
        return self.socket.recv()


class _RecordingSocket:
    def __init__(
        self,
        sender,
        *,
        block_data_recv=None,
        release_data_recv=None,
        fail_data_recv=False,
    ):
        self.sender = sender
        self.block_data_recv = block_data_recv
        self.release_data_recv = release_data_recv
        self.fail_data_recv = fail_data_recv
        self.messages = []
        self.recv_calls = 0

    def send_pyobj(self, value):
        message = dict(value) if isinstance(value, dict) else value
        if isinstance(message, dict) and "bucket_meta" in message:
            used = sum(meta["shape"].numel() * meta["dtype"].itemsize for meta in message["bucket_meta"].values())
            message["payload"] = self.sender.buffer[:used].clone()
        self.messages.append(message)

    def recv(self):
        self.recv_calls += 1
        if self.recv_calls == 2 and self.fail_data_recv:
            raise RuntimeError("receiver failed")
        if self.recv_calls == 2 and self.block_data_recv is not None:
            self.block_data_recv.set()
            assert self.release_data_recv.wait(timeout=5)
        return b"ok"

    def close(self):
        pass


class _PrefetchSender:
    bucket_size = 16
    bucket_size_mb = 0
    use_shm = False
    _mlite_prefetch_allow_cpu = True

    def __init__(self, **socket_kwargs):
        self.socket_kwargs = socket_kwargs
        self.cleaned = False
        self.original_calls = 0

    def _init_socket(self):
        self.socket = _RecordingSocket(self, **self.socket_kwargs)

    def _init_buffer(self):
        self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8)
        self.socket.send_pyobj(("fake-handle", ()))
        self.socket.recv()

    def _cleanup(self):
        self.cleaned = True

    def _direct_send_large_weight(self, name, weight):
        self.socket.send_pyobj(
            {
                "bucket_meta": {
                    name: {
                        "name": name,
                        "shape": weight.shape,
                        "dtype": weight.dtype,
                        "offset": 0,
                        "handle": ("direct", name),
                    }
                },
                "is_last": False,
            }
        )
        self.socket.recv()

    async def async_send_weights(self, weights):
        self.original_calls += 1
        return "original"


def _data_messages(sender):
    return [message for message in sender.socket.messages if isinstance(message, dict) and "bucket_meta" in message]


def _prefetch_sender_class():
    return type(
        "PrefetchSender",
        (_PrefetchSender,),
        {"async_send_weights": _PrefetchSender.async_send_weights},
    )


def test_prefetch_builds_third_bucket_while_first_ack_is_blocked():
    ack_blocked = threading.Event()
    release_ack = threading.Event()
    third_bucket_entered = threading.Event()
    fourth_bucket_entered = threading.Event()
    observed_third_bucket = threading.Event()
    observed_bounded_window = threading.Event()

    def weights():
        yield "a", torch.arange(4, dtype=torch.float32)
        yield "b", torch.arange(4, dtype=torch.float32) + 10
        third_bucket_entered.set()
        yield "c", torch.arange(4, dtype=torch.float32) + 20
        fourth_bucket_entered.set()
        yield "d", torch.arange(4, dtype=torch.float32) + 30

    sender_cls = _prefetch_sender_class()
    assert _install_bucketed_sender_prefetch(sender_cls)
    sender = sender_cls(block_data_recv=ack_blocked, release_data_recv=release_ack)

    def release_after_prefetch():
        assert ack_blocked.wait(timeout=5)
        if third_bucket_entered.wait(timeout=1):
            observed_third_bucket.set()
        if not fourth_bucket_entered.is_set():
            observed_bounded_window.set()
        release_ack.set()

    waiter = threading.Thread(target=release_after_prefetch)
    waiter.start()
    asyncio.run(sender.async_send_weights(weights()))
    waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert observed_third_bucket.is_set()
    assert observed_bounded_window.is_set()


def test_prefetch_allocates_exactly_two_staging_slots(monkeypatch):
    allocations = []
    original_empty_like = torch.empty_like

    def tracked_empty_like(tensor):
        result = original_empty_like(tensor)
        allocations.append(result)
        return result

    monkeypatch.setattr(torch, "empty_like", tracked_empty_like)
    sender_cls = _prefetch_sender_class()
    assert _install_bucketed_sender_prefetch(sender_cls)
    sender = sender_cls()

    asyncio.run(
        sender.async_send_weights(
            [(str(index), torch.arange(4, dtype=torch.float32)) for index in range(4)]
        )
    )

    assert len(allocations) == 2
    assert allocations[0].data_ptr() != allocations[1].data_ptr()


def test_prefetch_preserves_payloads_and_marks_partial_terminal_bucket():
    sender_cls = _prefetch_sender_class()
    assert _install_bucketed_sender_prefetch(sender_cls)
    sender = sender_cls()
    weights = [
        ("a", torch.tensor([1, 2], dtype=torch.float32)),
        ("b", torch.tensor([3, 4], dtype=torch.float32)),
        ("c", torch.tensor([5, 6], dtype=torch.bfloat16)),
    ]

    asyncio.run(sender.async_send_weights(weights))

    messages = _data_messages(sender)
    assert [list(message["bucket_meta"]) for message in messages] == [["a", "b"], ["c"]]
    assert [message["is_last"] for message in messages] == [False, True]
    assert messages[0]["payload"].tolist() == torch.cat(
        [weights[0][1].view(torch.uint8), weights[1][1].view(torch.uint8)]
    ).tolist()
    assert messages[1]["payload"].tolist() == weights[2][1].view(torch.uint8).tolist()
    assert sender.cleaned


def test_prefetch_empty_weights_and_producer_failure_cleanup():
    sender_cls = _prefetch_sender_class()
    assert _install_bucketed_sender_prefetch(sender_cls)
    empty_sender = sender_cls()
    asyncio.run(empty_sender.async_send_weights([]))
    assert [message["is_last"] for message in _data_messages(empty_sender)] == [True]
    assert empty_sender.cleaned

    def broken_weights():
        yield "a", torch.ones(1)
        raise RuntimeError("producer failed")

    broken_sender = sender_cls()
    with pytest.raises(RuntimeError, match="producer failed"):
        asyncio.run(broken_sender.async_send_weights(broken_weights()))
    assert broken_sender.cleaned


def test_prefetch_receiver_failure_stops_full_producer_queue_and_cleans_up():
    sender_cls = _prefetch_sender_class()
    assert _install_bucketed_sender_prefetch(sender_cls)
    sender = sender_cls(fail_data_recv=True)

    with pytest.raises(RuntimeError, match="receiver failed"):
        asyncio.run(
            sender.async_send_weights(
                [(str(index), torch.arange(4, dtype=torch.float32)) for index in range(8)]
            )
        )

    assert sender.cleaned


def test_prefetch_preserves_direct_send_for_oversized_weight():
    sender_cls = _prefetch_sender_class()
    assert _install_bucketed_sender_prefetch(sender_cls)
    sender = sender_cls()

    asyncio.run(
        sender.async_send_weights(
            [
                ("large", torch.arange(5, dtype=torch.float32)),
                ("small", torch.arange(1, dtype=torch.float32)),
            ]
        )
    )

    messages = _data_messages(sender)
    assert [list(message["bucket_meta"]) for message in messages] == [
        ["large"],
        ["small"],
    ]
    assert messages[0]["bucket_meta"]["large"]["handle"] == ("direct", "large")


def test_prefetch_falls_back_for_async_iterators_and_is_idempotent():
    sender_cls = _prefetch_sender_class()
    assert _install_bucketed_sender_prefetch(sender_cls)
    assert not _install_bucketed_sender_prefetch(sender_cls)

    async def weights():
        yield "a", torch.ones(1)

    sender = sender_cls()
    assert asyncio.run(sender.async_send_weights(weights())) == "original"
    assert sender.original_calls == 1


def test_prefetch_composes_with_probe_instrumentation(monkeypatch, capsys):
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE", "1")
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE_BACKEND", "prefetch-test")

    def fake_all_gather_into_tensor(output, tensor, *args, **kwargs):
        del args, kwargs
        output.copy_(tensor)

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor
    )
    sender_cls = _prefetch_sender_class()
    assert _install_bucketed_sender_prefetch(sender_cls)
    assert _instrument_bucketed_weight_sender(sender_cls)

    def weights():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output = torch.empty(1, device=device)
        torch.distributed.all_gather_into_tensor(output, torch.ones(1, device=device))
        yield "a", output

    sender = sender_cls()
    asyncio.run(sender.async_send_weights(weights()))

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("MLITE_WEIGHT_SYNC_PROBE ")
    )
    report = json.loads(line.split(" ", 1)[1])
    # IPC init plus one partial terminal data bucket.
    assert report["stages"]["handshake"]["calls"] == 4
    assert report["stages"]["mbridge_gather"]["calls"] == 1
    assert report["stages"]["mbridge_gather"]["bytes"] == 4


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_probe_flag_rejects_false_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MLITE_WEIGHT_SYNC_PROBE", raising=False)
    else:
        monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE", value)

    assert not _weight_sync_probe_enabled()


def test_sender_patch_installs_lazy_hook_without_importing_verl(monkeypatch):
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE", "1")
    monkeypatch.delitem(sys.modules, _BUCKETED_SENDER_MODULE, raising=False)
    original_meta_path = list(sys.meta_path)
    try:
        assert _patch_bucketed_weight_sender()
        assert _BUCKETED_SENDER_MODULE not in sys.modules
        assert not _patch_bucketed_weight_sender()
    finally:
        sys.meta_path[:] = original_meta_path


@pytest.mark.skipif(importlib.util.find_spec("verl") is None, reason="veRL is required")
def test_lazy_hook_patches_real_verl_sender(monkeypatch):
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE", "1")
    _patch_bucketed_weight_sender()
    module = importlib.import_module(_BUCKETED_SENDER_MODULE)
    assert module.BucketedWeightSender._mlite_weight_sync_probe_patch
    assert module.BucketedWeightSender._mlite_weight_prefetch_patch


def test_sender_patch_is_idempotent_and_profiles_handshake(monkeypatch, capsys):
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE", "1")
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE_BACKEND", "test-backend")

    assert _instrument_bucketed_weight_sender(_Sender)
    assert not _instrument_bucketed_weight_sender(_Sender)
    assert asyncio.run(_Sender().async_send_weights([])) == b"ok"

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("MLITE_WEIGHT_SYNC_PROBE ")
    )
    report = json.loads(line.split(" ", 1)[1])
    assert report["backend"] == "test-backend"
    assert report["stages"]["handshake"]["calls"] == 2
    assert report["stages"]["handshake"]["bytes"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sender_patch_profiles_real_host_to_device_copy(monkeypatch, capsys):
    class CudaSender:
        def _init_socket(self):
            self.socket = _Socket()

        async def async_send_weights(self, weights):
            self._init_socket()
            src = torch.arange(8, dtype=torch.float32)
            dst = torch.empty_like(src, device="cuda")
            dst.copy_(src)
            self.socket.send_pyobj({"is_last": True})
            self.socket.recv()
            return dst.cpu()

    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE", "1")
    monkeypatch.setenv("MLITE_WEIGHT_SYNC_PROBE_BACKEND", "cuda-test")
    assert _instrument_bucketed_weight_sender(CudaSender)

    result = asyncio.run(CudaSender().async_send_weights([]))
    torch.testing.assert_close(result, torch.arange(8, dtype=torch.float32))

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("MLITE_WEIGHT_SYNC_PROBE ")
    )
    report = json.loads(line.split(" ", 1)[1])
    assert report["stages"]["h2d"]["calls"] == 1
    assert report["stages"]["h2d"]["bytes"] == 32
