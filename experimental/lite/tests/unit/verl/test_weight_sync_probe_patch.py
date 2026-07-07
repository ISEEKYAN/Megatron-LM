# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys

import pytest
import torch

from verl_mlite.compat import (
    _BUCKETED_SENDER_MODULE,
    _instrument_bucketed_weight_sender,
    _patch_bucketed_weight_sender,
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
    assert _patch_bucketed_weight_sender()
    module = importlib.import_module(_BUCKETED_SENDER_MODULE)
    assert module.BucketedWeightSender._mlite_weight_sync_probe_patch


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
