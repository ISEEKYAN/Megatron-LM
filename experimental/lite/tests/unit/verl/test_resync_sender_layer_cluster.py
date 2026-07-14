import torch


def test_sync_bucket_producer_flushes_on_layer_boundary() -> None:
    from verl_mlite.compat import _SyncBucketProducer

    weights = [
        ("layers.0.attn.wq_a.weight", torch.zeros(8, dtype=torch.uint8)),
        ("layers.1.attn.wq_a.weight", torch.zeros(8, dtype=torch.uint8)),
    ]
    producer = _SyncBucketProducer(weights, bucket_size=32)
    staging = torch.empty(32, dtype=torch.uint8)

    kind, meta, _, used, _, is_last = producer.next_bucket(staging)

    assert kind == "bucket"
    assert set(meta) == {"layers.0.attn.wq_a.weight"}
    assert used == 8
    assert not is_last

    kind, meta, _, used, _, is_last = producer.next_bucket(staging)

    assert kind == "bucket"
    assert set(meta) == {"layers.1.attn.wq_a.weight"}
    assert used == 8
    assert is_last
