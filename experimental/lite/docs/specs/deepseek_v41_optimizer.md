# DeepSeek-V4.1-Flash optimizer contract

Status: specification and expected test vectors only. This document does not
implement an optimizer, enumerate a constructed training model, or certify training.

## Sources and evidence boundary

The source revision is `deepseek-ai/DeepSeek-V4.1-Flash` at
`df42c109f1defefcbfcedbe7d905718a12266e40`:

- [Technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/DeepSeek_V41_Tech_Report.pdf), §2.5, Algorithm 1 (printed pp. 15–16), §3.1.3, §4.2.2 (p. 22).
- [model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/model.py), especially lines 296–365, 438–550, 639–650, 793–826, 927–960, 1080–1130, 1220–1222.
- [vision.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/vision.py), lines 25–112.
- [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/config.json) and [checkpoint index](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/model.safetensors.index.json).

Local source bytes were inspected; web rendering was unavailable. SHA-256:

| Artifact | SHA-256 |
|---|---|
| Report | `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d` |
| model.py | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| vision.py | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` |
| engram.py | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| config.json | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| Checkpoint index | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |

The index contains 96,085 serialized keys, including quantization metadata and
2,401 `mtp.*` keys. A serialized key is not necessarily a trainable parameter.
Inference source establishes parameter roles and axes, not backward correctness
or the exact optimizer implementation used for training. Below, **R** means an
explicit report rule, **S** means its semantic lowering using source operations,
and **OPEN** means a decision the public material does not settle. OPEN entries
must not acquire a silent default when training is implemented.

## Official hyperparameters

| Quantity | Official value / interpretation | Source |
|---|---|---|
| AdamW | beta1=0.9, beta2=0.95, epsilon=1e-20; weight decay=0.1 | §4.2.2 |
| Muon | Nesterov momentum=0.95; decoupled weight decay=0.1 | §2.5, §4.2.2 |
| Muon update magnitude | Rescale each logical update matrix to RMS=0.18; reuse AdamW LR | §4.2.2 |
| Sinkhorn | Nesterov momentum=0.95; K=11 alternating normalization steps, tau=1e-3, epsilon=1e-20 | Algorithm 1, §4.2.2 |
| Sinkhorn LR correction | gamma=0.18, applied once after sqrt(n) update scaling | §2.5, Algorithm 1 |
| Sinkhorn decay | 0, including embedding and prediction head | §2.5 |
| Engram LR | 5× base LR; exact scope within Engram is OPEN below | §4.2.2 |
| Normalization weights | Weight decay=0.1, including semantically normalization weights stored as matrices | §2.5 + S |
| Biases and scaling factors | Weight decay=0 | §2.5 |
| Text / image router correction bias | Update speed=0.001 for each modality | §4.2.2 |
| Sequence-level balance auxiliary loss | Coefficient=0.0001 | §4.2.2 |

The optimizer K=11 is unrelated to mHC's `hc_sinkhorn_iters=20` and
`hc_eps=1e-6`. RMSNorm epsilon=1e-20 is also a separate consumer.

The reported pretraining schedule is a 2,000-step linear warmup to 2.6e-4,
constant until 28T tokens, cosine decay to 2.6e-5 over 28T–40T, then constant
through 45T; batch size is 100.6M tokens. Sequence length starts at 64K and
extends to 1M at 34T. These are pretraining facts, not imposed RL defaults.
The vision encoder is frozen until LR decay, except its final norm; the aligner
also remains trainable. After unfreezing the encoder has a smaller LR whose
numeric multiplier is not specified here by the report (OPEN).

## Semantic routing manifest

Names use the official checkpoint namespace. `L=layers.{0..39}`, `E` denotes
an individual expert, and `V=vision.blocks.{0..31}`. Patterns below are semantic
families, not a first-match regular-expression implementation. All matrices
refer to dequantized logical weights in [output, input] orientation. TP/EP/DP
shards, padding, flattened optimizer buffers, and FP4 packed bytes do not define
logical matrices. Multiple physical aliases of one parameter have one owner.

LR multipliers are relative to the base schedule and exclude the 0.18 update
correction. `e_proj` and `e_norm` are OPEN (1 versus 5 requires a recorded
choice); Engram table LR is 5. `v` is the OPEN smaller encoder LR after unfreeze.
An inactive parameter gets no optimizer state or decay.

| Parameter name | Algorithm | Logical matrix / head split | LR multiplier | Weight decay | Basis |
|---|---|---|---|---|---|
| `embed.weight`, `head.weight` | Sinkhorn | Each [129280,5120] token-feature matrix separately; embeddings are untied | 1 | 0 | R/S |
| `L.engram.embed.weight` (L=1,14) | Sinkhorn | [384006168,256] or [384016682,256]; whole table statistics, not independently per hash head | 5 | 0 | R/S |
| `L.engram.wkv.weight` | Muon | [25600,6144], linear projection; no attention-head split | e_proj | decoupled 0.1 | R/S; LR OPEN |
| `L.engram.q_weight`, `L.engram.k_weight` | AdamW | [4,5120] elementwise normalization gains per HC copy; NOT linear Q/K projections | e_norm | 0.1 | R/S; LR OPEN |
| `L.attn.wq_a.weight` | Muon | [1280,5120] shared low-rank projection, one matrix | 1 | decoupled 0.1 | S |
| `L.attn.wq_b.weight` | head-wise Muon | [64,512,1280], 64 independent [512,1280] matrices | 1 | decoupled 0.1 | R/S |
| `L.attn.wkv.weight` | Muon | [512,5120], ONE shared latent K/V head, never replicate into 64 query heads | 1 | decoupled 0.1 | S |
| `L.attn.indexer.wq_b.weight` | head-wise Muon | [32,128,1280], 32 independent [128,1280] matrices | 1 | decoupled 0.1 | R/S |
| `L.attn.indexer.wk.weight` | Muon | [128,512], one shared index key matrix | 1 | decoupled 0.1 | R/S |
| `L.attn.indexer.weights_proj.weight` | Muon | [32,5120], predicts head scores; not a Query projection | 1 | decoupled 0.1 | S |
| `L.attn.compressor.wkv.weight`, `.wgate.weight` | Muon | Each [512,5120] projection separately; wgate only where instantiated | 1 | decoupled 0.1 | S |
| `L.attn.wo_a.weight` | Muon | Forward uses [8,1024,4096]: 8 grouped linear maps; proposed per-group lowering, exact training grouping OPEN | 1 | decoupled 0.1 | S/OPEN |
| `L.attn.wo_b.weight` | Muon | [5120,8192], one matrix | 1 | decoupled 0.1 | S |
| `L.ffn.experts.E.w1/w3.weight`, `L.ffn.shared_experts.w1/w3.weight` | Muon | Each [2304,5120] separately; never mix experts or gate/up projections | 1 | decoupled 0.1 | R/S |
| `L.ffn.experts.E.w2.weight`, `L.ffn.shared_experts.w2.weight` | Muon | Each [5120,2304] separately | 1 | decoupled 0.1 | R/S |
| `L.ffn.gate.weight` | Muon | [384,5120] linear routing scores, not correction bias | 1 | decoupled 0.1 | S |
| `L.hc_attn_fn`, `L.hc_ffn_fn` | Muon | Each [24,20480] linear map, no head split | 1 | decoupled 0.1 | S |
| `L.hc_attn_base`, `L.hc_ffn_base`, `L.hc_attn_scale`, `L.hc_ffn_scale` | AdamW | Additive bias vectors / multiplicative scaling factors | 1 | 0 | R/S |
| `L.attn.attn_sink` | AdamW | [64] attention-logit bias | 1 | 0 | S |
| `L.attn_norm.weight`, `L.ffn_norm.weight`, `L.attn.q_norm.weight`, `L.attn.kv_norm.weight`, `L.attn.compressor.norm.weight`, `L.attn.indexer.k_norm.weight`, `norm.weight` | AdamW | Elementwise RMSNorm gains of their declared feature dimensions | 1 | 0.1 | R/S |
| `L.ffn.gate.bias`, `L.ffn.gate.bias_vl` | Auxiliary-loss-free correction update, NOT AdamW | Separate text/image expert-load vectors [384]; update speed 0.001 each | N/A | 0 | R/S |
| `vision.patch_embed.proj.weight` | Muon when active | [1024,588] linear map | v | decoupled 0.1 | R/S |
| `V.attn.wqkv.weight` | Muon when active | [3072,1024] splits into contiguous Q,K,V [1024,1024]; Q and K each split into 16 [64,1024] heads; V remains one [1024,1024] matrix | v | decoupled 0.1 | R/S |
| `V.attn.wo.weight` | Muon when active | [1024,1024] | v | decoupled 0.1 | R/S |
| `V.mlp.w1.weight` | Muon when active | Fused [5632,1024] gate/up; separate [2816,1024] maps proposed, exact fused training grouping OPEN | v | decoupled 0.1 | S/OPEN |
| `V.mlp.w2.weight` | Muon when active | [1024,2816] | v | decoupled 0.1 | R/S |
| `V.norm1.weight`, `V.norm2.weight` | AdamW when active | [1024] normalization gains | v | 0.1 | R/S |
| `vision.norm.weight` | AdamW, active before encoder unfreeze | [1024] normalization gains | 1 before unfreeze; later OPEN | 0.1 | R/S |
| `vision.patch_embed.proj.bias`, `V.attn.wqkv.bias`, `V.attn.wo.bias` | AdamW when active | Bias vectors, including fused QKV bias | v | 0 | R/S |
| `aligner.w1.weight`, `aligner.w2.weight` | Muon | [5120,9216] and [5120,5120] | 1 | decoupled 0.1 | R/S |
| `aligner.w1.bias`, `aligner.w2.bias` | AdamW | [5120] biases | 1 | 0 | R/S |
| `image_start`, `image_end`, `image_newline` | OPEN: embedding semantics suggest Sinkhorn, non-matrix storage suggests AdamW | Each [5120]; whether packed as special-token rows is not specified | OPEN | OPEN | Evidence gap; no default |
| Quantized `*.scale` accompanying serialized weights | No independent optimizer group in the proposed master-weight training representation | Quantization metadata, regenerated with quantized values; unlike learned `hc_*_scale` | N/A | N/A | S, training representation must confirm |
| `mtp.*` | Excluded / frozen for this integration | All 2401 checkpoint keys retained by the weight-loading contract; no DSpark forward or training | N/A | No update | Scope |

The report's wording does not resolve every fused/grouped layout or the scope
of the Engram multiplier. These OPEN fields are specification evidence gaps,
not official numeric claims. A future training configuration must resolve them
explicitly before creating active groups. The table does not authorize guessing
an indexer training-loss coefficient or a router-load reduction scope.

The MoE correction biases influence top-k selection, not the unbiased routing
weights. Their 0.001 update rate is not a gradient LR; the 0.0001 sequence loss
coefficient belongs to the training objective, not an optimizer group. Training
must define modality-specific load aggregation, empty-modality handling and
update timing from a separate training reference.

## Algorithm 1 reference contract

For each logical token-feature matrix W of shape [m,n], start with M_previous
(zero at initialization). Use beta=0.95, tau=1e-3, epsilon=1e-20, K=11:

```text
M = beta * M_previous + (1-beta) * G
N = beta * M + (1-beta) * G
rho[i] = L2(N[i,:]); rho_mean = sum(rho) / m
U = copy(N)                         # restart from CURRENT N every optimizer step
U[i,:] = 0 where rho[i] <= tau * rho_mean
for k in 1..11:
    if k is odd: U[i,:] /= L2(U[i,:]) + epsilon
    else:        U[:,j] /= L2(U[:,j]) + epsilon
Delta = sqrt(n) * U
W_next = W - (0.18 * base_lr * group_lr_multiplier) * Delta
```

The mask is computed once from N before normalization, using all m rows,
including zero rows. Epsilon is added AFTER the L2 norm, not under its square
root. There are six row normalizations and five column normalizations, ending
with rows. Finite K does not guarantee exact column RMS=1; tests must compare
the algorithm, not impose exact doubly balanced output. All-zero input and
momentum produce a zero update even for nonzero W because decay is absent.

For table sharding, the row L2 norm must combine hidden-axis shards before the
mask; rho_mean requires all logical rows; column norms require all logical
rows. Padding and duplicated replicas must not count as extra rows. Momentum
state sharding across replicas does not change this mathematical matrix.

§3.1.3 describes retaining row/column scaling vectors across iterations to
avoid full-matrix writes. This permits a candidate implementation of the same
11-step computation; it does NOT establish optimizer-step warm-start semantics.
Only M is required persistent state by Algorithm 1. A scaling-vector cache is
optional and must reproduce a fresh-N reference for each step, including changed
masks, zero columns and resumed execution. No cache may silently replace U(0).

## Expected test cards (implementation-stage obligations)

These cards specify fixtures and mutations; they are not claims that a model or
optimizer implementation has passed. Exact statements below use real arithmetic.
For floating comparisons, use an independent scalar float64 Algorithm 1 reference
with fixed reduction order and epsilon placement; establish error bounds and
obtain numerical-threshold review before accepting distributed/low-precision
results. No undefined “BF16 tolerance” or inherited bitwise claim is authorized.

| Card | Reference and fixture | Expected check | Mutation and why it fails |
|---|---|---|---|
| S1 zero / no decay | W=ones([2,2]), Mprev=G=zeros, eta=1 | M=N=Delta=0; Wnext=W exactly | Applying 0.1 decay changes W despite zero update |
| S2 signed symmetric | Mprev=0, G=[[1,-1],[-1,1]], eta=1 | M=0.05G; N=0.0975G; in float64 Delta approximately G; Wnext approximately W-0.18G; Engram table approximately W-0.9G | Missing sqrt(n), double gamma, or omitted 5× changes magnitude |
| S3 momentum continuity | First G=ones([2,2]), then G=0 | M1=0.05 ones; N1=0.0975 ones; M2=0.0475 ones; N2=0.045125 ones; second update remains nonzero | Resetting M each step yields zero second update; unnormalized momentum convention disagrees with M1 |
| S4 epsilon location | Mprev=0; choose scalar G=1e-20/0.0975 so N=1e-20 | First row-normalized U is 0.5 in real arithmetic; inspect this intermediate | sqrt(sum(U²)+epsilon) instead gives approximately 1e-10 |
| S5 threshold equality | Inject N=[1,1999,0,0]^T with m=4,n=1 | rho_mean=500; first row survives (threshold 0.5); other zero rows masked | Rank-local mean on rows [1,1999] is 1000 and wrongly masks row 1 |
| S6 inclusive threshold | Inject N=[1,1999]^T | rho_mean=1000; first row masked at equality rho=1 | Using strict less-than preserves first row |
| S7 normalization order | Inject N=[[1,2],[3,4],[0,0]], retain every U(k) | Match independent scalar reference at all 11 stages; third row remains zero | K=10, column-first, or masking after normalization disagrees with a stage reference |
| S8 cache semantics | Same current W,Mprev,G as S7, two arbitrary prior scaling caches (ones versus nonuniform positive vectors) | Both results equal fresh-N reference; changing the previous step's mask cannot change current mask | Multiplying N by stale scaling vectors before fixed K steps generally changes finite-K result |
| R1 semantic Q/K | wq_a [1280,5120], wq_b [32768,1280], indexer Q [4096,1280], indexer K [128,512], latent KV [512,5120] | Matrix counts 1,64,32,1,1 with shapes in manifest | Substring q/k classification splits shared low-rank projections or invents key heads |
| R2 fused vision | Mark each Q/K/V block and each of 16 heads with distinct sentinel values | 16 Q matrices + 16 K matrices + 1 V matrix, reconstruct original row order exactly | Applying 48-head split changes V; interleaved QKV interpretation changes sentinels |
| R3 semantic normalization | Engram q_weight/k_weight [4,5120], hc_*_scale [3], norm.weight [5120] | AdamW for all; decay 0.1,0,0.1 respectively | ndim>=2→Muon misroutes Engram gains; blanket ndim==1 no-decay misroutes norm |
| R4 unknown / ownership | Add a trainable mystery.weight, then introduce alias and frozen MTP fixture | Unknown rejected; aliases update once; MTP has no state/update | Catch-all AdamW conceals missing roles; counting checkpoint keys confuses metadata with parameters |
| R5 correction biases | Supply separate text/image load vectors with opposite overloaded experts | Distinct modality bias updates at speed 0.001 under approved load-update reference; neither bias is in gradient optimizer | Collapsing bias and bias_vl couples independent routing decisions; AdamW applies an unrelated update |
| R6 distributed matrix | S5 split across row ranks; S7 split by columns; add padded rows and replicas | Same logical mask and full-reference update after gathering, within reviewed numeric bounds | Rank-local statistics or replica double counting alter mask/preconditioner |
| R7 lifecycle | Skip step, save/load, state offload/onload, then repeat S3 | Atomic skip across algorithms and biases; resumed result equals uninterrupted reference; momentum tensors actually move | Adam-only state-key whitelist silently leaves momentum resident; config-only type checks miss SGD fallback |

Post-construction acceptance must enumerate actual trainable parameter objects
and prove exactly one justified group per logical owner, plus explicit exclusions.
It must reject unresolved active OPEN entries and unknown parameters. Actual
constructor types, state tensors, clipping, skip-step, checkpoint continuity,
freeze/unfreeze initialization and offload must be checked through the real
mixed-optimizer path. Existing generic optimizer wrapping does not establish
this routing contract. In particular, Muon offload must not silently select SGD
or move only Adam state keys. These are later implementation gates, not gates
for this specification. A CPU fixture is not full-scale training evidence.

The S7 rounded, 70-digit decimal reference is stored in
[deepseek_v41_optimizer_vectors.json](deepseek_v41_optimizer_vectors.json).
It injects N after momentum so normalization can be tested independently.
It was calculated on CPU with Python `decimal`, using `Context.prec=70`,
`sqrt(sum(x*x)) + Decimal('1e-20')`, masking once, then eleven alternating
row/column divisions and final multiplication by `Decimal(2).sqrt()`.
This is a mathematical reference vector, not an approved floating tolerance.

Specification verification inspected the pinned source/configuration and counted
96,085 index keys (2,401 MTP). CPU decimal checks verified S1's zero update,
S3's momentum intermediates, S4's epsilon placement, S5/S6's masks, and generated
S7. No training model was instantiated; no GPU or distributed test was run.
