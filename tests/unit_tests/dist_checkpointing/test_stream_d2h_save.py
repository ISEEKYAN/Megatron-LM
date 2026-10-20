# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Synchronous save that stages device tensors one item at a time and drops written pages."""

import os

import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.strategies import filesystem_async
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestStreamedD2HSave:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_streamed_save_round_trips_without_staging_the_shard(
        self, tmp_path_dist_ckpt, monkeypatch
    ):
        """Device tensors reach the writer unstaged, round-trip bitwise, and pages are dropped."""
        Utils.initialize_model_parallel(1, 1)
        big = torch.arange(64 * 64, dtype=torch.float32, device="cuda").reshape(64, 64) + Utils.rank
        # Replicated across ranks: every rank holds the same values.
        small = torch.randn(3, 5, generator=torch.Generator().manual_seed(0)).to(
            "cuda", torch.bfloat16
        )
        state = {
            "big": ShardedTensor.from_rank_offsets("big", big, (0, Utils.rank, Utils.world_size)),
            "small": ShardedTensor.from_rank_offsets("small", small, replica_id=Utils.rank),
        }
        # 1 KiB buffer: `small` reuses it, `big` (16 KiB) takes the oversized path.
        monkeypatch.setenv("MCORE_DIST_CKPT_STREAM_D2H", "1")
        monkeypatch.setenv("MCORE_DIST_CKPT_STAGE_BYTES", "1024")
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

        with TempNamedDir(tmp_path_dist_ckpt / "test_stream_d2h") as ckpt_dir:
            save(state, ckpt_dir, async_sharded_save=False)
            assert preloaded and all(t.is_cuda for t in preloaded)
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
