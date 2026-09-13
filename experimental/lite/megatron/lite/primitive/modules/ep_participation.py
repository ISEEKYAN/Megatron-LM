# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded host-side rendezvous before native EP collectives.

Guards host participation only, not CUDA completion. A failed group must be torn
down. Calls on a group must be serialized and identically ordered across ranks,
exactly like the EP collectives they precede.
"""

from __future__ import annotations

import json
import time
from weakref import WeakKeyDictionary

import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_process_group_store

_TIMEOUT = 10.0


class _Participation:
    def __init__(self, store, ranks, rank):
        self.store = store
        self.ranks = ranks
        self.keys = [str(r) for r in ranks]
        self.key = str(rank)
        self.sequence, self.phase, self.failure, self.ready = 0, "", None, False

    def _records(self):
        keys = self.keys if self.ready else [k for k in self.keys if self.store.check([k])]
        return dict(zip(keys, map(json.loads, self.store.multi_get(keys))))

    def check(self, phase):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        self.sequence += 1
        sequence = self.sequence
        self.store.set(self.key, json.dumps([sequence, phase, self.phase]))
        self.phase = phase
        deadline = time.monotonic() + _TIMEOUT
        while True:
            # Never Store.get an absent key: that inherits the store's own
            # (1800s) timeout. check() is non-blocking, one key per rank.
            self.ready = self.ready or self.store.check(self.keys)
            expired = time.monotonic() >= deadline
            if not self.ready and not expired:
                time.sleep(0.001)
                continue
            records = self._records()
            missing, mismatched = [], []
            for rank, key in zip(self.ranks, self.keys):
                count, current, previous = records.get(key, [0, "", ""])
                if count < sequence:
                    missing.append(f"rank {rank} expected={sequence} actual={count}")
                    continue
                # A peer may already be waiting at the next rendezvous.
                actual = current if count == sequence else previous
                if count > sequence + 1 or actual != phase:
                    mismatched.append(
                        f"rank {rank} expected={sequence}:{phase} actual={count}:{actual}"
                    )
            if mismatched or (missing and expired):
                self.failure = (
                    f"EP participation failed before {phase}: "
                    f"expected participants={len(self.ranks)} "
                    f"actual={len(self.ranks) - len(missing)}; " + "; ".join(missing + mismatched)
                )
                raise RuntimeError(self.failure)
            if not missing:
                return sequence
            time.sleep(0.001)


_participation = WeakKeyDictionary()


def check_ep_participation(group, phase):
    """Require every EP peer within 10s, assuming the rendezvous Store is live."""
    if group is None:
        group = dist.group.WORLD
    if dist.get_world_size(group) <= 1:
        return None
    state = _participation.get(group)
    if state is None:
        # Reuse the group's namespace: no extra collective or group creation on
        # a rank-local dispatch path, so an absent rank need not run this code.
        store = dist.PrefixStore("mlite_ep_participation", _get_process_group_store(group))
        state = _Participation(store, dist.get_process_group_ranks(group), dist.get_rank())
        _participation[group] = state
    return state.check(phase)
