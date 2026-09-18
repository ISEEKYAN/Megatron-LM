# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unsupported parallel combinations are rejected by a table, before any setup.

The guards used to be scattered ``raise`` statements inside ``build_model``.
They are now a declarative table plus one validator loop, so these tests pin the
three properties that make the table equivalent to what it replaced: every entry
is reachable, each raises its own error type with its own named message, and the
whole table is evaluated before the model or the process group is touched.
"""

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite.protocol import UNSUPPORTED, build_model


class _Parallel:
    def __init__(self, **kwargs):
        self.tp = self.vpp = self.pp = self.ep = self.cp = 1
        self.etp = None
        self.pp_layout = None
        self.__dict__.update(kwargs)


class _Impl:
    def __init__(self, **kwargs):
        self.text_only = True
        self.external_vision_device = None
        self.pipeline_split_layer = 20
        self.optimizer = None
        self.parallel = _Parallel()
        self.__dict__.update(kwargs)


def _build(**kwargs):
    return build_model(object(), impl_cfg=_Impl(**kwargs))


# --- the coarse dimension check, which runs before the table ----------------


@pytest.mark.parametrize(
    "parallel, named",
    [
        (_Parallel(tp=2), "tp"),
        (_Parallel(vpp=2), "vpp"),
        (_Parallel(pp=4), "pp"),
        (_Parallel(etp=2), "etp"),
        (_Parallel(pp_layout=[[0], [1]]), "pp_layout"),
    ],
)
def test_unsupported_dimensions_are_named_individually(parallel, named):
    with pytest.raises(NotImplementedError) as caught:
        _build(parallel=parallel)
    message = str(caught.value)
    assert message.startswith("V4.1_UNSUPPORTED_PARALLELISM:")
    # The offending knob is named, not just "unsupported": an operator reading
    # the log must be able to tell which setting to change.
    assert named in message.split(";")[0]


def test_several_unsupported_dimensions_are_reported_together():
    with pytest.raises(NotImplementedError) as caught:
        _build(parallel=_Parallel(tp=2, vpp=2))
    head = str(caught.value).split(";")[0]
    assert "tp" in head and "vpp" in head


# --- the declarative table --------------------------------------------------


def test_every_table_entry_is_well_formed():
    assert len(UNSUPPORTED) >= 4
    for invalid, error, message in UNSUPPORTED:
        assert callable(invalid)
        assert isinstance(error, type) and issubclass(error, Exception)
        assert message.strip(), "a guard with no message is unactionable"


def test_the_documented_condition_keys_all_survive_the_table_rewrite():
    # The guards moved from scattered ``raise`` statements into this table; the
    # documented keys are what operators grep for in a job log, so losing one is
    # a silent regression even though the condition may still be rejected.
    # (Four further entries carry prose messages with no key -- e.g. "EP size
    # must be a positive integer" -- preserved verbatim from the statements they
    # replaced rather than renamed, so they are deliberately not required here.)
    keys = {message.partition(":")[0] for _, _, message in UNSUPPORTED}
    assert {
        "V4.1_PP_TEXT_ONLY",
        "V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED",
        "V4.1_PP_COMBINATION_UNSUPPORTED",
        "V4.1_PP_OPTIMIZER_UNSUPPORTED",
        "CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED",
    } <= keys


@pytest.mark.parametrize(
    "kwargs, key",
    [
        (dict(text_only=False), "V4.1_PP_TEXT_ONLY"),
        (dict(external_vision_device="cuda:0"), "V4.1_PP_TEXT_ONLY"),
        (dict(pipeline_split_layer=10), "V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED"),
        (dict(optimizer="muon"), "V4.1_PP_OPTIMIZER_UNSUPPORTED"),
    ],
)
def test_pp2_rejections_name_their_own_cause(kwargs, key):
    with pytest.raises(NotImplementedError) as caught:
        _build(parallel=_Parallel(pp=2), **kwargs)
    assert str(caught.value).startswith(key), str(caught.value)


@pytest.mark.parametrize("extra", [dict(ep=2), dict(cp=2)])
def test_pp2_rejects_being_combined_with_ep_or_cp(extra):
    with pytest.raises(NotImplementedError) as caught:
        _build(parallel=_Parallel(pp=2, **extra))
    assert str(caught.value).startswith("V4.1_PP_COMBINATION_UNSUPPORTED")


def test_split_layer_20_is_the_one_accepted_cut():
    # The topology table puts every KV owner (2/8/14/20) on the same stage as
    # its readers only when the cut falls on layer 20; the guard encodes that.
    predicate = next(
        invalid
        for invalid, _, message in UNSUPPORTED
        if message.startswith("V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED")
    )
    parallel = _Parallel(pp=2)
    assert not predicate(_Impl(pipeline_split_layer=20), parallel)
    for cut in (2, 8, 14, 19, 21, 39):
        assert predicate(_Impl(pipeline_split_layer=cut), parallel), cut
    # With PP off the cut is irrelevant and must not be rejected.
    assert not predicate(_Impl(pipeline_split_layer=10), _Parallel(pp=1))


def test_guards_run_before_any_distributed_setup(monkeypatch):
    # A guard that only fires after init_parallel would hang the ranks that do
    # reach it and never reject on the ranks that do not.
    def explode(*args, **kwargs):
        raise AssertionError("distributed setup ran before the guards rejected")

    monkeypatch.setattr(torch.distributed, "is_initialized", explode)
    with pytest.raises(NotImplementedError):
        _build(parallel=_Parallel(tp=2))
    with pytest.raises(NotImplementedError):
        _build(parallel=_Parallel(pp=2), text_only=False)
