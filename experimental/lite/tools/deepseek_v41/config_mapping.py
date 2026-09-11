"""Explicit release-leaf to official ModelArgs mapping, with closed scope waivers."""

TEXT_FIELDS = {
    "vocab_size": "vocab_size",
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "head_dim": "head_dim",
    "qk_rope_head_dim": "rope_head_dim",
    "q_lora_rank": "q_lora_rank",
    "o_lora_rank": "o_lora_rank",
    "o_groups": "o_groups",
    "swiglu_limit": "swiglu_limit",
    "rms_norm_eps": "norm_eps",
    "rope_theta": "rope_theta",
    "rope_scaling.factor": "rope_factor",
    "rope_scaling.beta_fast": "beta_fast",
    "rope_scaling.beta_slow": "beta_slow",
    "rope_scaling.original_max_position_embeddings": "original_seq_len",
    "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts",
    "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func",
    "norm_topk_prob": "norm_topk_prob",
    "routed_scaling_factor": "route_scale",
    "sliding_window": "window_size",
    "compress_ratios": "compress_ratios",
    "compress_rope_theta": "compress_rope_theta",
    "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "candidate_source_layer_id": "candidate_source_layer",
    "candidate_topk_blocks": "candidate_topk_blocks",
    "candidate_block_size": "candidate_block_size",
    "hc_mult": "hc_mult",
    "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "hc_eps": "hc_eps",
    "engram_layer_ids": "engram_layer_ids",
    "engram_num_embeddings": "engram_num_embeddings",
    "engram_max_ngram_size": "engram_max_ngram_size",
    "engram_vocab_size": "engram_vocab_size",
    "engram_n_heads": "engram_n_heads",
    "engram_head_dim": "engram_head_dim",
    "engram_pad_token_id": "engram_pad_id",
    "engram_compressed_vocab_size": "engram_compressed_vocab_size",
    "num_nextn_predict_layers": "n_mtp_layers",
    "dspark_block_size": "dspark_block_size",
    "dspark_noise_token_id": "dspark_noise_token_id",
    "dspark_target_layer_ids": "dspark_target_layer_ids",
    "dspark_markov_rank": "dspark_markov_rank",
    "dspark_n_routed_experts": "dspark_n_routed_experts",
    "dspark_num_experts_per_tok": "dspark_n_activated_experts",
}
VISION_FIELDS = {
    "num_hidden_layers": "vision_n_layers",
    "hidden_size": "vision_dim",
    "num_attention_heads": "vision_n_heads",
    "intermediate_size": "vision_inter_dim",
    "patch_size": "vision_patch_size",
    "rope_theta": "vision_rope_theta",
    "downsample_ratio": "vision_downsample_ratio",
    "max_image_tokens": "vision_max_n_token",
    "min_pixels": "vision_min_pixels",
    "max_wh_ratio": "vision_max_wh_ratio",
}
MAPPING = {
    **{"text_config." + k: v for k, v in TEXT_FIELDS.items()},
    **{"vision_config." + k: v for k, v in VISION_FIELDS.items()},
    "image_token_id": "image_token_id",
    "quantization_config.quant_method": "dtype",
    "quantization_config.expert_dtype": "expert_dtype",
}
WAIVERS = {
    "architectures": "archival dispatch metadata",
    "model_type": "archival dispatch metadata",
    "transformers_version": "archival version metadata",
    "text_config.model_type": "archival dispatch metadata",
    "vision_config.model_type": "archival dispatch metadata",
    "bos_token_id": "no tokenizer/generation in reduced input profile",
    "eos_token_id": "no generation in forward-only profile",
    "pad_token_id": "unpadded sequence boundary; padding is excluded",
    "dtype": "BF16 residual context and explicit FP32 head",
    "quantization_config.activation_scheme": "dynamic activation quantizer",
    "quantization_config.weight_block_size": "fixed 32x32 kernel ABI",
    "quantization_config.scale_fmt": "fixed UE8M0 kernel ABI",
    "text_config.num_key_value_heads": "single shared KV vector ABI",
    "text_config.hidden_act": "official SiLU implementation",
    "text_config.attention_bias": "official no-bias Linear contract",
    "text_config.attention_dropout": "inference has no dropout",
    "text_config.initializer_range": "every parameter overwritten by complete fixture weights",
    "text_config.use_cache": "official stateful cache execution",
    "text_config.tie_word_embeddings": "independent embedding and head bindings",
    "text_config.max_position_embeddings": "release upper bound; explicit runtime cache limit",
    "text_config.rope_scaling.rope_type": "official YaRN equations",
    "text_config.topk_method": "official correction-bias routing",
}


def leaves(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(leaves(item, name))
        else:
            result[name] = item
    return result


def map_release_config(release, defaults):
    flat = leaves(release)
    if set(flat) != set(MAPPING) | set(WAIVERS):
        raise ValueError("release config leaf coverage mismatch")
    output = dict(defaults)
    for source, target in MAPPING.items():
        if target not in output:
            raise ValueError(f"unknown ModelArgs target: {target}")
        output[target] = flat[source]
    return output
