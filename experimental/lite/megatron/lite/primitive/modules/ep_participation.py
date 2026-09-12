# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded host-side rendezvous before native EP collectives."""

from __future__ import annotations

import json
import time
from weakref import WeakKeyDictionary

import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_process_group_store


class _Participation:
    def __init__(self, group):
        # Reuse the group's namespace; no extra collective/group creation on a
        # rank-local dispatch path. An absent rank need not execute this code.
        self.store = dist.PrefixStore("mlite_ep_participation", _get_process_group_store(group))
        self.ranks = dist.get_process_group_ranks(group)
        self.keys = [str(rank) for rank in self.ranks]
        self.key = str(dist.get_rank())
        self.sequence = 0
        self.phase = ""
        self.failure = None
        self.ready = False

    def check(self, phase):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        self.sequence += 1
        sequence = self.sequence
        self.store.set(self.key, json.dumps([sequence, phase, self.phase]))
        self.phase = phase
        deadline = time.monotonic() + 10.0
        while True:
            # Never get a nonexistent key: Store.get otherwise waits for its
            # own (possibly 1800s) timeout. Keys remain bounded at one per rank.
            self.ready = self.ready or self.store.check(self.keys)
            if not self.ready and time.monotonic() < deadline:
                time.sleep(0.001)
                continue
            available = (
                self.keys if self.ready else [key for key in self.keys if self.store.check([key])]
            )
            records = dict(zip(available, map(json.loads, self.store.multi_get(available))))
            missing, mismatched = [], []
            for rank, key in zip(self.ranks, self.keys):
                count, current, previous = records.get(key, [0, "", ""])
                if count < sequence:
                    missing.append(f"rank {rank} expected={sequence} actual={count}")
                else:
                    # A peer may already be waiting at the next rendezvous.
                    actual_phase = current if count == sequence else previous
                    if count > sequence + 1 or actual_phase != phase:
                        mismatched.append(
                            f"rank {rank} expected={sequence}:{phase} actual={count}:{actual_phase}"
                        )
            if mismatched or (missing and time.monotonic() >= deadline):
                self.failure = (
                    f"EP participation failed before {phase}: "
                    f"expected participants={len(self.ranks)} "
                    f"actual={len(self.ranks) - len(missing)}; " + "; ".join(missing + mismatched)
                )
                raise RuntimeError(self.failure)
            if not missing:
                return
            time.sleep(0.001)


_participation = WeakKeyDictionary()


def check_ep_participation(group, phase):
    """Require every EP peer within 10s, assuming the rendezvous Store is live.

    This guards host participation, not CUDA completion or arbitrary Store
    failure. A failed group must be torn down. Like EP collectives themselves,
    calls on a group must be serialized and identically ordered across ranks.
    """
    if group is None:
        group = dist.group.WORLD
    if dist.get_world_size(group) <= 1:
        return
    state = _participation.get(group)
    if state is None:
        state = _Participation(group)
        _participation[group] = state
    state.check(phase)
