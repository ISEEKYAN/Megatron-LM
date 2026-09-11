import pytest

from megatron.lite.model.deepseek_v41.lite.parallel import EngramLayout


def test_existing_ranks_form_row_and_replica_groups():
    layout = EngramLayout(7, ((0, 2, 4), (1, 3, 5)), world_size=6)
    assert layout.row_intervals == ((0, 3), (3, 5), (5, 7))
    assert layout.replica_groups == ((0, 1), (2, 3), (4, 5))
    assert [layout.owner(r, replica=1) for r in range(7)] == [1, 1, 1, 3, 3, 5, 5]
    assert [layout.optimizer_interval(r) for r in range(6)] == [
        (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    assert layout.canonical_owner(3) == 2


def test_empty_row_and_optimizer_shards_cover_once():
    layout = EngramLayout(1, ((2, 0), (3, 1)), world_size=4)
    assert layout.row_intervals == ((0, 1), (1, 1))
    assert sorted(i for rank in range(4) for i in range(*layout.optimizer_interval(rank))) == [0]


@pytest.mark.parametrize("groups", [((0, 0),), ((0, 1), (2,)), ((-1, 0),), ((0, 6),), ()])
def test_invalid_rank_maps_fail(groups):
    with pytest.raises(ValueError):
        EngramLayout(7, groups, world_size=6)


def test_alias_and_shadow_resolve_once():
    layout = EngramLayout(7, ((0, 1),), world_size=2,
                          aliases={"reuse": "shadow", "shadow": "owner"})
    assert layout.canonical_name("reuse") == "owner"
    with pytest.raises(ValueError):
        EngramLayout(7, ((0, 1),), world_size=2, aliases={"a": "b", "b": "a"})
