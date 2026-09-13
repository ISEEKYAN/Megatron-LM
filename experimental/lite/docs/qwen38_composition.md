# Qwen3.8-Flash-Next text training composition

Single-rank MLite runtime training is assembled; distributed training, full HF loading, and end-to-end reference parity remain unvalidated.
References: [HF config and checkpoint index](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540), [Automodel PR](https://github.com/NVIDIA-NeMo/Automodel/pull/3690) head `5cfe13b160eb7e23ac5a4868bbf611707cdf98fb`; source inspection also used Automodel `dc8f31f2c35e9e98a8721b575037a833807c7ac1`.
Fresh source checks: Automodel main `f7ccd6f7902634af34c2f31b3294ac250dc97670` and PR head fetched 2026-09-13T11:12:06Z; MCore `nv/dev` `0cd11658f44350a141656751259cfe1f72398e9f` fetched 2026-09-13T11:11:23Z has generic GDN/HC, no dedicated Qwen3.8 model.
The original 2000-added-line component budget is exceeded by runtime assembly and training tests; no budget acceptance is claimed. No tools or fixtures are added.

| Config group | Mapping / invariant |
|---|---|
| Identity | `qwen4_exp` / `qwen4_exp_text`; unwrap `text_config`, retain vision config separately. |
| Decoder | H=2560, 48 layers, explicit `layer_types`: 36 GDN + 12 QSA; full-attention interval 4. |
| GDN | QK heads=16, V heads=48, head dims=128, convolution width=4; output gate is sigmoid. |
| QSA/RoPE | Q=24, KV=2, head dim=256, rotary dim=64, default theta=10000 (checkpoint config may override); index Q=4/K=1, dim=128, compression=4, token budget=2048. |
| MoE/HC | 512 experts, top-10, expert/shared width=640; four persistent streams, read bottleneck=320. |
| PLE | `ple_layer_ids=[2]` is one-based (decoder index 1); 16 heads of width 160, 128 physical shards, padded rows=320001536. |

| Operator | Selection / semantic difference |
|---|---|
| GDN recurrence/projections | Reuse GatedDeltaNet mechanism; extend gate selection to sigmoid and bypass Qwen3.5 pre-normalization after HC read. |
| MoE | Reuse routing/dispatch/experts and Qwen3.5 tensor transforms; HC replaces ordinary residual/pre-norm wiring. |
| HC | Reuse pipeline expand/fold/unfold only. New grouped RMSNorm, low-rank sigmoid read, mean over streams, and `2*sigmoid` injection write; reject DS4 mixing equations. |
| QSA | New compressed-block indexer and selected-token GQA; group keys before norm/RoPE, restart blocks per document, preserve causal tail. Reject DS4 CSA/DSA routing semantics. |
| N-gram | New raw-ID, EOS-aware signed-int64 multiply/XOR/modulo; reject DS4 tokenizer compression and seeded multipliers. Constants: 23703573157769, 20109073645365, 8052911324071. |
| PLE | Temporary model-local floating lookup; duplicate IDs accumulate gradients. New branch norms/projections and depthwise convolution (width 4, dilation 3, nine-token history). |
| CP | Reuse contiguous slicing, THD metadata/unpacking, router replay; retain global IDs and document boundaries, local interval `[r*L,(r+1)*L)`. Per-document CP slicing is not global contiguous slicing. |
| Storage | Use floating trainable owner rows for Qwen; reuse block-FP8 helpers only if a matching quantized storage contract is explicitly selected. Do not force DS4 FP8+E8M0 table semantics. |

Checkpoint index: 1658 keys = 1294 text/head + 333 vision + 31 MTP. Below, `L=model.language_model.layers.i`; mapping plans follow config/reference equations. Real-HF tests additionally check 7 HC native shapes, 9 GDN planned shapes, and all 35 PLE buffer integers; this is not a complete loader.
| HF keys (suffixes grouped only when transform is identical) | Proposed native mapping |
|---|---|
| `model.language_model.embed_tokens.weight`, `lm_head.weight` | Token embedding/output, each [248320,2560]; preserve untied weights. |
| `L.linear_attn.in_proj_{qkv,z,b,a}.weight` | Existing segmented Qwen3.5 merge/split and TP head transforms; preserve qkv,z,b,a order. |
| `L.linear_attn.{conv1d.weight,A_log,dt_bias,norm.weight,out_proj.weight}` | Existing GDN parameter placement; verify norm-weight convention and sigmoid behavior independently. |
| `L.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight`, `{q_norm,k_norm}.weight` | QSA projections; Q contains per-head query+gate, not plain Q. Q projection [12288,2560], K/V [512,2560]. |
| `L.self_attn.indexer.{index_qk_proj.weight,q_layernorm.weight,k_layernorm.weight}` | New QSA indexer: fused [640,2560] projection and two [128] norms; reference freezes indexer, main attention remains differentiable. |
| `L.mlp.experts.{gate_up_proj,down_proj}` | Grouped expert tensors, no `.weight` suffix; existing expert-axis/TP transforms, no global EP materialization. |
| `L.mlp.gate.weight`, `shared_expert.{gate_proj,up_proj,down_proj}.weight`, `shared_expert_gate.weight` | Existing router and shared-expert transforms; preserve independent shared gate. |
| `L.{attn_hyper_connection,mlp_hyper_connection}.{hc_norm,input_mix_weight_down,input_mix_weight_up,block_inject_weight}.weight` | New HC state; widths 10240, down [320,10240], up [10240,320], inject [4,10240]. |
| `model.language_model.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}.weight` | Final read-only HC mixer; no injected branch or invented final norm. |
| `L.ple.ple_embedding.{layer_multipliers,ngram_heads_vocab_sizes,ngram_heads_offsets}` | Preserve checkpoint buffers; compare all three constants and all 16 primes/offsets with independent expected values. |
| `L.ple.ple_embedding.ngram_embedding.shard_j.weight` | 128 shards, each expected [2500012,160]; map intersections directly into resident owner rows, never concatenate global table. |
| `L.ple.{key_proj,value_proj}.weight`, `{norm_key,norm_query,norm_conv}.weight`, `conv1d.weight` | Qwen PLE: key [10240,2560], value [2560,2560], norms [10240], convolution [10240,1,4]. |
| `model.visual.*` (333), `mtp.*` (31) | Account explicitly in coverage; current MLite text composition does not establish their support. Reference adapter omits MTP; this is not evidence of complete checkpoint coverage. |

Engram is explicitly temporary and independent of PR #212: after that PR lands on main, a separate follow-up will converge on `primitive/modules/engram_lookup.py`. Qwen uses floating rows/raw token IDs; DS4 also supports FP8/scales and tokenizer compression. Multi-owner lookup is currently rejected.
Keep `contiguous_slice_for_cp(tensor, cp_rank, cp_size, seq_dim=1)`, `thd_pack_meta(seq_lens, *, tp_size=1, cp_size=1, cp_group=None, contiguous=False)`, and `unpack_thd_to_nested(output, meta, *, contiguous=False)` stable. Confirm global packed slicing separately from per-document padding.
The registered protocol uses the existing MLite runtime and distributed optimizer for single-rank scratch training. Set `load_hf_weights=False`; HF import/export and all parallel dimensions greater than one fail explicitly. The GPU test exercises GDN, QSA, MoE, HC, PLE, optimizer updates and model-state restoration; packed GDN requires FLA. No full skill acceptance is claimed.
