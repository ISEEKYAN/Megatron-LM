# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import asyncio
import json

from verl_mlite.compat import _instrument_bucketed_weight_sender


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
