import torch


def test_math_prompts_are_fixed_and_cover_at_least_32_cases() -> None:
    from examples.verl.ds4_resync_tp4 import math_prompts

    prompts = math_prompts()
    assert len(prompts) >= 32
    assert len(set(prompts)) == len(prompts)
    assert all(prompt.startswith("Solve briefly:") for prompt in prompts)


def test_distribution_comparison_reports_kl_and_selected_token_delta() -> None:
    from examples.verl.ds4_resync_tp4 import compare_distributions

    reference = [
        {
            "logprobs": torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), -1),
            "token_ids": torch.tensor([2]),
        }
    ]
    candidate = [
        {
            "logprobs": torch.log_softmax(torch.tensor([[1.1, 1.9, 3.0]]), -1),
            "token_ids": torch.tensor([2]),
        }
    ]
    report = compare_distributions(reference, candidate)
    assert report["token_count"] == 1
    assert report["max_abs"] > 0
    assert report["max_kl"] > 0
    assert report["max_selected_token_logprob_delta"] >= 0


def test_copy_checkpoint_metadata_recurses_but_excludes_weights(tmp_path) -> None:
    from examples.verl.ds4_resync_tp4 import copy_checkpoint_metadata

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text("{}")
    (source / "inference").mkdir()
    (source / "inference" / "model.py").write_text("class Model: pass\n")
    (source / "model.safetensors").write_bytes(b"weights")
    copy_checkpoint_metadata(source, output)
    assert (output / "config.json").read_text() == "{}"
    assert (output / "inference" / "model.py").read_text() == "class Model: pass\n"
    assert not (output / "model.safetensors").exists()


def test_model_vocab_size_includes_non_tokenizer_slots() -> None:
    from examples.verl.ds4_resync_tp4 import model_vocab_size

    class Config:
        vocab_size = 129280

    class Tokenizer:
        vocab_size = 128000

    assert model_vocab_size(Config(), Tokenizer()) == 129280
