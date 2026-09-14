# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded overlapping CP/EP ownership contract.

Automodel@8a646a739, mesh_utils._create_moe_mesh, flattens DP/CP before
reshaping EP. Its PLE lookup uses row owners; the subsequent convolution halo
uses sequence owners. MLite retains these two adjoint routes and uses native
expert-DP buffers for local row/expert weights, instead of FSDP DTensors.
Only the CP2/EP2, DP1/expert-DP1 contract is admitted here.
"""

import torch.distributed as dist


def validate_cp_ep_contract(ps, *, ple_owner_sharding):
    if not ple_owner_sharding or any(
        getattr(ps, field) != expected
        for field, expected in (
            ('cp_size', 2),
            ('ep_size', 2),
            ('dp_size', 1),
            ('expert_dp_size', 1),
            ('tp_size', 1),
            ('etp_size', 1),
            ('pp_size', 1),
        )
    ):
        raise NotImplementedError('QWEN38_CP_COMBINATION_NOT_VALIDATED')
    groups = [ps.cp_group, ps.ep_group, ps.dp_cp_group, ps.ep_dp_group]
    if any(group is None for group in groups):
        raise ValueError('QWEN38_CP_EP_OWNER_GROUPS')
    cp, ep, dense, expert = [dist.get_process_group_ranks(g) for g in groups]
    if (
        len(cp) != 2
        or cp != ep
        or cp != dense
        or ps.cp_rank not in (0, 1)
        or ps.ep_rank != ps.cp_rank
        or expert != [cp[ps.cp_rank]]
    ):
        raise ValueError('QWEN38_CP_EP_OWNER_GROUPS')
