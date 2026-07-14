from verl_mlite.rollout.layer_cluster import (
    LayerClusterBuffer,
    iter_layer_clustered_weights,
    resync_layer_cluster_key,
)


def test_resync_layer_cluster_key_groups_ds4_weight_and_scale() -> None:
    weight = "layers.12.attn.wq_a.weight"
    scale = "layers.12.attn.wq_a.scale"
    assert resync_layer_cluster_key(weight) == (1, 12)
    assert resync_layer_cluster_key(scale) == resync_layer_cluster_key(weight)


def test_resync_layer_cluster_key_handles_qwen_and_top_level_tensors() -> None:
    assert resync_layer_cluster_key("model.language_model.layers.3.mlp.experts.gate_up_proj") == (
        1,
        3,
    )
    assert resync_layer_cluster_key("embed.weight") == (0, 0)
    assert resync_layer_cluster_key("head.weight") == (4, 0)


def test_iter_layer_clustered_weights_groups_consecutive_layer_runs() -> None:
    stream = [
        ("layers.0.attn.wq_a.weight", 0),
        ("layers.0.attn.wq_a.scale", "scale0"),
        ("layers.1.attn.wq_a.weight", 1),
        ("layers.1.attn.wq_a.scale", "scale1"),
    ]
    clustered = list(iter_layer_clustered_weights(stream))
    assert [name for name, _ in clustered] == [name for name, _ in stream]


def test_layer_cluster_buffer_flushes_on_boundary() -> None:
    loads: list[list[str]] = []
    buffer = LayerClusterBuffer(lambda batch: loads.append([name for name, _ in batch]))
    buffer.ingest_bucket([("layers.0.a", 1), ("layers.1.a", 2)])
    buffer.ingest_bucket([("layers.1.b", 3)])
    buffer.finalize()
    assert loads == [["layers.0.a"], ["layers.1.a", "layers.1.b"]]
