"""Optional frozen-token FP32-definition probe; never a production indexer.

The CPU reference does not reproduce the old DeepGEMM reduction order. Its
Top-K/logprob deltas describe that reference, not a bitwise W7 native replay.
"""

import os
from collections import defaultdict
from functools import wraps

import torch


def fp32_score_reference(query, keys, weights, lengths):
    from megatron.lite.primitive.modules.deployment_math.cache import IndexCache
    from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4

    def decoded_cpu(value):
        if isinstance(value, IndexCache):
            return dequantize_mxfp4(
                value.packed.detach().cpu().view(torch.int8),
                value.scale.detach().cpu().view(torch.float8_e8m0fnu),
            )
        return value.detach().float().cpu()

    q, k, w = (decoded_cpu(value) for value in (query, keys, weights))
    bounds = torch.broadcast_to(lengths.detach().cpu(), q.shape[:2] + (1,))
    scores = q.new_empty(q.shape[:2] + (k.shape[1],))
    columns = torch.arange(k.shape[1])
    for start in range(0, q.shape[1], 128):
        stop = start + 128
        chunk = (
            torch.einsum('bshd,btd->bsht', q[:, start:stop], k).relu()
            * w[:, start:stop].unsqueeze(-1)
        ).sum(2)
        scores[:, start:stop] = chunk.masked_fill(
            ~(columns < bounds[:, start:stop]), -torch.inf
        )
    return scores


def install(event):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.modules import deployment_math as dm

    original = protocol._forward_step_impl
    if getattr(original, '_ds41_indexer_diagnostic', False):
        return
    counts = defaultdict(int)
    limit = int(os.environ.get('DS41_INDEXER_DIAGNOSTIC_LIMIT', '1'))

    @wraps(original)
    def forward(model, batch, **kwargs):
        result = original(model, batch, **kwargs)
        # Re-entering forward is unsafe under PP: the receiving stage has
        # consumed its carrier. This diagnostic is valid only for PP1.
        if model.ps.pp_size > 1 or model.ps.cp_size > 1:
            return result
        if (
            model.training
            or torch.is_grad_enabled()
            or not batch.input_ids.is_cuda
            or len(model.layers) != 4
            or 'log_probs' not in result
            or counts[id(model)] >= limit
        ):
            return result
        counts[id(model)] += 1
        versions = [parameter._version for parameter in model.parameters()]
        native = result['log_probs'].detach().clone()
        score_records = []
        native_scores = dm.indexer_scores

        def reference_scores(query, keys, weights, lengths):
            reference = fp32_score_reference(query, keys, weights, lengths).to(
                query.device
            )
            actual = native_scores(query, keys, weights, lengths)
            selected = dm.indexer_select_positions(actual, lengths, 512)
            alternate = dm.indexer_select_positions(reference, lengths, 512)
            truncated = (
                torch.broadcast_to(lengths, query.shape[:2] + (1,))[..., 0] > 512
            )
            changed = (selected != alternate).any(-1)
            valid = torch.arange(keys.shape[1], device=query.device) < lengths
            error = (actual - reference).abs().masked_fill(~valid, 0)
            score_records.append(
                {
                    'rows': query.shape[0] * query.shape[1],
                    'columns': keys.shape[1],
                    'truncated_rows': int(truncated.sum().item()),
                    'changed_topk_rows': int((changed & truncated).sum().item()),
                    'max_score_abs_diff': error.max().item() if error.numel() else 0.0,
                    'input_weights_dtype': str(weights.dtype),
                }
            )
            return reference

        probe_kwargs = dict(kwargs, optimizer=None)
        with torch.random.fork_rng(devices=[batch.input_ids.device.index]):
            dm.indexer_scores = reference_scores
            try:
                alternate = original(model, batch, **probe_kwargs)
            finally:
                dm.indexer_scores = native_scores
            replay = original(model, batch, **probe_kwargs)
        assert versions == [parameter._version for parameter in model.parameters()]
        assert torch.equal(
            native, replay['log_probs']
        ), 'Diagnostic changed the native replay'
        mask = protocol._cp_targets(batch, None)[1].bool()
        difference = (native - alternate['log_probs']).abs()[mask]
        assert difference.numel() and torch.isfinite(difference).all()
        assert score_records and sum(row['truncated_rows'] for row in score_records) > 0
        event(
            'indexer_fp32_definition_probe',
            {
                'reference': 'CPU FP32 definition; not old DeepGEMM bitwise arithmetic',
                'role': 'actor' if kwargs.get('optimizer') is not None else 'ref',
                'probe_ordinal': counts[id(model)],
                'score_calls': score_records,
                'response_tokens': difference.numel(),
                'response_logprob_max_abs_diff': difference.max().item(),
                'response_logprob_mismatches': int((difference != 0).sum().item()),
                'native_replay_equal': True,
                'parameter_versions_unchanged': True,
            },
        )
        return result

    forward._ds41_indexer_diagnostic = True
    protocol._forward_step_impl = forward
