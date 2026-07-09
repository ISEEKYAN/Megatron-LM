# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit process-group ownership for the standalone M-FSDP primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class MFSDPProcessGroups:
    dense_dp: dist.ProcessGroup | None
    expert_dp: dist.ProcessGroup | None
    dense_ag: dist.ProcessGroup | None
    expert_ag: dist.ProcessGroup | None
    tp: dist.ProcessGroup | None
    etp: dist.ProcessGroup | None
    ep: dist.ProcessGroup | None
    pp: dist.ProcessGroup | None

    def data_group(self, *, expert: bool) -> dist.ProcessGroup | None:
        return self.expert_dp if expert else self.dense_dp

    def gather_group(self, *, expert: bool) -> dist.ProcessGroup | None:
        return self.expert_ag if expert else self.dense_ag

    def registration_groups(self) -> tuple[dist.ProcessGroup, ...]:
        groups: list[dist.ProcessGroup] = []
        for group in (self.dense_dp, self.expert_dp, self.dense_ag, self.expert_ag):
            if group is not None and all(group is not existing for existing in groups):
                groups.append(group)
        return tuple(groups)


def build_mfsdp_process_groups(ps: Any) -> MFSDPProcessGroups:
    """Read groups already owned by MLite; never initialize global MCore state."""
    dense_dp = getattr(ps, "dp_cp_group", None) or getattr(ps, "dp_group", None)
    expert_dp = getattr(ps, "ep_dp_group", None) or dense_dp
    dense_ag = getattr(ps, "dp_cp_ag_group", None) or getattr(ps, "dp_ag_group", None)
    expert_ag = getattr(ps, "ep_dp_ag_group", None)
    return MFSDPProcessGroups(
        dense_dp=dense_dp,
        expert_dp=expert_dp,
        dense_ag=dense_ag or dense_dp,
        expert_ag=expert_ag or expert_dp,
        tp=getattr(ps, "tp_group", None),
        etp=getattr(ps, "etp_group", None),
        ep=getattr(ps, "ep_group", None),
        pp=getattr(ps, "pp_group", None),
    )


def group_size(group: dist.ProcessGroup | None) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def group_rank(group: dist.ProcessGroup | None) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group)


__all__ = [
    "MFSDPProcessGroups",
    "build_mfsdp_process_groups",
    "group_rank",
    "group_size",
]
