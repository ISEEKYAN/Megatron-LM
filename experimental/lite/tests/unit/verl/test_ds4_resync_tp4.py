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
