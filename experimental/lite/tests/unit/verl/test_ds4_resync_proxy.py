def test_proxy_impl_config_is_concrete_and_uses_local_attention() -> None:
    from examples.verl.ds4_resync_proxy import _proxy_impl_config

    impl = _proxy_impl_config()
    parallel = impl.parallel

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
    assert impl.attention_backend_override == "local"
    assert impl.optimizer is None
    assert impl.mtp_enable is False
