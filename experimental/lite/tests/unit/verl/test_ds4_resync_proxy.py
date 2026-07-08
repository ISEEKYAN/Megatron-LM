def test_proxy_parallel_dimensions_are_fully_concrete() -> None:
    from examples.verl.ds4_resync_proxy import _proxy_parallel_config

    parallel = _proxy_parallel_config()

    assert (
        parallel.tp,
        parallel.etp,
        parallel.ep,
        parallel.pp,
        parallel.vpp,
        parallel.cp,
    ) == (
        1,
        1,
        1,
        1,
        1,
        1,
    )
