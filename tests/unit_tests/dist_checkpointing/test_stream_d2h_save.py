# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Synchronous save that stages device tensors one item at a time and drops written pages."""
import errno
import io
import os

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.strategies import filesystem_async
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestStreamedD2HSave:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("stream", [True, False])
    def test_save_round_trips_and_drops_written_pages(
        self, tmp_path_dist_ckpt, monkeypatch, stream
    ):
        """Streamed: device tensors reach the writer unstaged. Both modes round-trip bitwise
        through the EFAULT-retry writer and drop written pages."""
        Utils.initialize_model_parallel(1, 1)
        big = torch.arange(64 * 64, dtype=torch.float32, device="cuda").reshape(64, 64) + Utils.rank
        small = (torch.arange(15, device="cuda").reshape(3, 5) / 7).bfloat16()
        state = {
            "big": ShardedTensor.from_rank_offsets("big", big, (0, Utils.rank, Utils.world_size)),
            "small": ShardedTensor.from_rank_offsets("small", small, replica_id=Utils.rank),
        }
        monkeypatch.setenv("MCORE_DIST_CKPT_STREAM_D2H", "1" if stream else "0")
        monkeypatch.setenv("MCORE_DIST_CKPT_DROP_CACHE_BYTES", "4096")
        preloaded, dropped = [], []
        preload = FileSystemWriterAsync.preload_tensors

        def spy_preload(write_buckets, non_blocking=True):
            out = preload(write_buckets, non_blocking)
            preloaded.extend(t for _, _, (_, tensors) in out for _, t in tensors)
            return out

        fadvise = os.posix_fadvise
        monkeypatch.setattr(FileSystemWriterAsync, "preload_tensors", staticmethod(spy_preload))
        monkeypatch.setattr(
            filesystem_async.os,
            "posix_fadvise",
            lambda fd, off, n, advice: (dropped.append(advice), fadvise(fd, off, n, advice)),
        )

        with TempNamedDir(tmp_path_dist_ckpt / f"test_stream_d2h_{stream}") as ckpt_dir:
            save(state, ckpt_dir, async_sharded_save=False)
            assert preloaded and all(t.is_cuda == stream for t in preloaded)
            assert dropped and set(dropped) == {os.POSIX_FADV_DONTNEED}
            loaded = load(
                {
                    "big": ShardedTensor.from_rank_offsets(
                        "big", torch.empty_like(big), (0, Utils.rank, Utils.world_size)
                    ),
                    "small": ShardedTensor.from_rank_offsets(
                        "small", torch.empty_like(small), replica_id=Utils.rank
                    ),
                },
                ckpt_dir,
            )
        assert torch.equal(loaded["big"], big)
        assert torch.equal(loaded["small"].view(torch.int16), small.view(torch.int16))


class _FailOnce(io.BytesIO):
    """Writes part of the second 4-byte chunk, then fails once with ``err``."""

    name = "fail-once"

    def __init__(self, err):
        super().__init__()
        self.err = err

    def write(self, data):
        if self.err and self.tell() >= 4:
            err, self.err = self.err, None
            super().write(bytes(data)[:2])
            raise OSError(err, os.strerror(err))
        return super().write(data)


def test_efault_rewrites_the_chunk_from_its_start():
    raw = _FailOnce(errno.EFAULT)
    assert filesystem_async._EfaultRetryStream(raw, chunk_bytes=4).write(b"abcdefghij") == 10
    assert raw.getvalue() == b"abcdefghij"


def test_other_write_errors_are_not_retried():
    stream = filesystem_async._EfaultRetryStream(_FailOnce(errno.EIO), chunk_bytes=4)
    with pytest.raises(OSError) as raised:
        stream.write(b"abcdefgh")
    assert raised.value.errno == errno.EIO
