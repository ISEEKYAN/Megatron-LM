# DeepSeek-V4.1 independent training mathematics

Specification for `tests/reference/deepseek_v41/training.py` and
`test_training_reference.py`; neither implementation nor training acceptance.
The reference must use scalar/float64 arithmetic, without importing native MLite
operators, DS4 backward code, inference autograd wrappers or target optimizers.
Expected cards below are analytical inputs and results, not captured target output.

## Provenance and scope

Report: [pinned technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/DeepSeek_V41_Tech_Report.pdf),
SHA256 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.
Local report bytes were checked against this digest. Relevant sections are
2.1 (modality routing), 2.2 equation 1 (CED), 2.3 (CSA2), 2.4.1 equation 2
(mHC), 2.5 Algorithm 1 (updates), 3.1.2 (sharing gradients), 3.1.3 (Engram),
4.2.2 (optimizer constants), and 5 (post-training scope).

Consume these immutable prerequisites at repository-relative paths; branches need
not already be merged. They take precedence over older prose:

| Source | Commit | Contract consumed |
|---|---|---|
| Corrected A1 | `2b2c7e0c324be7df47f66fe18be393e53bef2fee` | `experimental/lite/docs/deepseek_v41_owner_consumer.md`: paired h20/p20; 40 layers and owners |
| A1/A3 correction | `9d1697a28` | `experimental/lite/docs/specs/deepseek_v41_csa2.md`: latent branching, rotary and compressor order |
| A2 | `b3a916425` | `experimental/lite/docs/contracts/deepseek_v41/`: semantic weights; inactive 2,401 MTP keys |
| A4 | `c4c27b0e6` | `experimental/lite/docs/specs/deepseek_v41_optimizer{.md,_vectors.json}`: groups and S1–S8/R1–R7 |
| A8 | `766bc22d1396c30a6c7d08deabe64f0a56e84d8f` | `experimental/lite/docs/plans/deepseek_v41_{decisions.json,a8_evidence.md}`: approved post-training policies |
| B3-S | `43ac7cf7f` | `experimental/lite/docs/specs/deepseek_v41_fixtures.md` and `deepseek_v41_fixture_vectors.json`: dimensions, margins, ties, masks |

R below means explicit report rule; D is mathematical differentiation of the
fixed floating graph; P is approved port policy, not official training evidence.
A diagnostic scalar objective is not a substitute for an unknown model objective.
No DSpark forward, rollout or MTP gradient is included.

## Reference interface and precision

Each call receives a manifest with source revisions, profile, trainable owner IDs,
parameter values, masks, sequence-local positions, fixed discrete maps, objective
numerator/denominator, runtime scale ledger and active decisions. Return named
intermediates, scalar numerator and denominator, scalar loss, input VJPs, owner
parameter gradients and optional next optimizer states/deltas. Preserve absent
versus zero gradients: frozen, unexecuted and disconnected have distinct reasons.
Unknown owners and active unresolved decisions are errors before execution.

Use BSH plus explicit HC/head dimensions from B3; no implicit shape reduction.
Float64 CPU algebra comparisons use atol=rtol=1e-12. Central differences use
h=1e-5 and atol=rtol=1e-8 on the well-conditioned smooth cards below. Test h/2
as a stability check. Near-singular norms need separate conditioning analysis.
Integer selections, owner identity, masks and exclusions require exact equality.
These bounds do not approve BF16/FP8/FP4 or distributed/native kernel tolerance.
O10 requires native FP32 main_grad production, accumulation, storage and reduction;
casting a BF16 gradient to FP32 is not equivalent. O11 stores momentum in FP32.

## Differentiable graph

Write bar(x) for a cotangent and use row-vector linear convention y=x W^T.
All expressions below are D unless explicitly marked R/P.

* Linear: bar(x)=bar(y) W; bar(W)=sum_tokens bar(y)^T x.
* RMSNorm: r=(mean(x^2)+eps)^(-1/2), y=g*x*r;
  bar(x)=r*(g*bar(y))-x*r^3*mean(x*g*bar(y));
  bar(g)=sum_tokens bar(y)*x*r. Keep eps inside this square root.
* Softmax: p=softmax(a), bar(a)=p*(bar(p)-sum(p*bar(p))).
* RoPE at a fixed position: y=R x, bar(x)=R^T bar(y); no position gradient.
* SwiGLU: y=SiLU(a)*b, bar(a)=bar(y)*b*sigmoid(a)*
  (1+a*(1-sigmoid(a))), bar(b)=bar(y)*SiLU(a).
  Where the pinned forward clamps activations, differentiate that clamp only in
  its smooth interior/exterior; record boundary convention separately.

For ratio-2 compression, per feature, c_j=sum_t alpha_jt*v_jt,
alpha_j=softmax(z_j) over the two tokens of the completed group.
bar(v_jt)=alpha_jt*bar(c_j),
bar(z_jt)=alpha_jt*(v_jt-c_j)*bar(c_j).
Then backpropagate both projections and norm. Ratio 1 has projection and norm,
no gate or softmax. Incomplete groups are invisible; packed samples restart.
Forward projection precision follows A3, not the float64 diagnostic dtype.

At owner 20 use u=sum_a p20[a]*h20[a], x20=attn_norm_20(u),
c20=compressor_norm_20(W20*x20) in column notation (A1 correction).
bar(h20[a]) += p20[a]*bar(u); bar(p20[a]) += dot(h20[a],bar(u)).
The paired coefficient p20 stays in the graph. Report equation 1 is conceptual;
it does not authorize raw-h20 projection, an extra decoder projection, or detach.
Index K is projected from c20 before main RoPE; main KV and index K subsequently
have their separate rotary/quantization paths.

For a fixed selected set J and each query/head, concatenate local SWA and selected
global entries without deduplication. With logits a_j=q·k_j/sqrt(d), sink s,
p=softmax([a,s]), o=sum_j p_j*v_j (sink value zero):
bar(a_j)=p_j*dot(bar(o),v_j-o),
bar(s)=-p_sink*dot(bar(o),o), bar(v_j)=p_j*bar(o),
bar(q)=sum_j bar(a_j)*k_j/sqrt(d), bar(k_j)=bar(a_j)*q/sqrt(d).
Where K and V share one latent, add BOTH cotangents before its backward.
Causal masks exclude entries from both normalization and gradients.

Report 3.1.2 requires one logical owner and aggregation from all consumers (R):
bar(c_owner)=sum_consumers bar(c_from_consumer), including its own layer.
Backpropagate the owner producer once after summation; shadow parameter gradients
sum into the same owner once. Integer Top-K/candidate maps and hashes have no
cotangents. Reindex can produce fresh scores; Reuse executes no index query.
Fixed-selection LM loss alone therefore does not train the indexer scoring path.
It must not manufacture an STE for Top-K or silently supply an auxiliary loss.

For mHC equation 2, Y=B X+C f(A X) (R), let Z=A X and F=f(Z):
bar(B)=bar(Y) X^T; bar(C)=bar(Y) F^T; bar(F)=C^T bar(Y);
bar(Z)=J_f^T bar(F); bar(A)=bar(Z) X^T;
bar(X)=B^T bar(Y)+A^T bar(Z)+J_H^T[bar(A),bar(B),bar(C)].
The last term cannot be omitted. Differentiate the actual finite 20-iteration
HC normalization, not an infinite doubly stochastic projection or the optimizer's
11-iteration Sinkhorn. Single-pass scheduling follows A1 paired state.

Engram lookup at fixed integer IDs is gather; backward is scatter-add including
all repeated rows and consumers (R 3.1.3). It gives no token/hash gradient.
Differentiate floating projections, norms and gates by the chain rule; trainable
quantized table storage/update remains conditional on O08/O09.
Pixel unshuffle is a permutation whose VJP is its inverse; repeat of image vectors
into four HC slots has VJP equal to the sum of four slot cotangents. Image masks,
nonaligned spans and packed boundaries are B3 rules, not differentiable indices.

Quantization backward is P/O14: FQ(x)=x+stop_gradient(Q(x)-x), with scales detached.
Its VJP is identity in x, including saturation. Q uses each distinct A3 codec.
A finite difference of quantized forward will not recover this surrogate VJP;
test forward codec and custom backward separately. Never use this test to claim
the true quantizer is smooth.

## Objectives and runtime scaling

The executable baseline contract is next-token CE with explicit supervision mask,
y_t the next token within the same sample, z_t all-token logits,
n=sum_t m_t*(logsumexp(z_t)-z_t[y_t]), D=sum_t m_t, L=n/D (P/O16).
bar(z_t)=m_t*(softmax(z_t)-onehot(y_t))/D. No cross-sample label shift.
Mask choice is supplied by the SFT/RL/OPD caller, not inferred from text_mask:
image tokens may affect subsequent supervised text through the floating graph.
D=0 requires an explicit no-objective result with zero cotangents; the runtime
must apply its approved skip policy before any optimizer/scheduler publication.

For microbatches r, accumulate numerator gradients and divide by sum_r D_r.
A mean of microbatch means is wrong when D_r differ. If the runtime averages
across P replicas and accumulates A microbatch means, explicitly record every
factor and compensate so the final gradient equals sum_r grad(n_r)/sum_r D_r.
Do not blindly multiply by P or A; the ledger must describe the actual reducer.

Report 4.2.2 gives sequence-balance coefficient 0.0001 (R), but the complete
per-sequence/per-modality objective and its post-training activation must come
from a reviewed recipe. That coefficient alone cannot define the loss.
Report 2.1 keeps modality correction biases separate and out of routing weights
(R); O13 inherits the DS4 TP count reduction separately per modality (P).
Neither an auxiliary-loss coefficient nor bias update speed is a gradient LR.
RL advantage construction, clipping, KL and OPD teacher/detach choices are
caller objectives and require their own explicit contracts; CE cards certify none.

## Update contract

Use A4 Algorithm 1 literally: M=.95*Mprev+.05*G; N=.95*M+.05*G;
mask rows with norm(N_i)<=.001*mean_row_norm(N), initialize U from current N,
then 11 alternating row/column divisions by (L2 norm+1e-20), row first and last.
Delta=sqrt(n)*U; Wnext=W-.18*eta*group_multiplier*Delta; no decay (R).
Only momentum persists mathematically; old scaling caches cannot change Delta.
A4 S1–S8, including the independent 70-digit S7 vector, remain required.

For AdamW, explicit step t starts at 1, m=.9*m_prev+.1*g,
v=.95*v_prev+.05*g^2, mh=m/(1-.9^t), vh=v/(1-.95^t),
Wnext=(1-eta*lambda)*W-eta*mh/(sqrt(vh)+1e-20).
This is the declared conventional mathematical card using A4 constants;
backend epsilon, step count and bias correction must match before native parity.
Muon cards must declare the polar/orthogonalization approximation and zero-update
rule; report RMS=.18 and momentum=.95 do not uniquely specify a finite iterative
backend. A8 inherits DS4 grouping, not an invented new grouped lowering.
Conditional Engram projections/norms use 5x only when trainable (O01/O02).
Aliases update once; MTP has no gradient, optimizer state, decay or update.
Atomic skip follows O17 and requires a real backend audit downstream.

## Independent expected cards

All small values are exact real arithmetic unless a tolerance is stated.
These cards do not replace B3's full 40-layer owner manifest.

| ID | Inputs and scalar objective | Independent expectation / discriminating failure |
|---|---|---|
| T1 owner accumulation | c=2*x, x=3; L=3*c+5*c | c=6, L=48, bar(c)=8, bar(x)=16; omit, duplicate or detach either consumer fails |
| T2 CED pair | h=[2,5], p=[1/4,3/4], u=sum(p*h), L=2*u | u=17/4, L=17/2, bar(h)=[1/2,3/2], bar(p)=[4,10]; diagnostic collapse before norm |
| T3 pooling | v=[2,6], z=[0,0], L=softmax(z)·v | L=4, bar(v)=[1/2,1/2], bar(z)=[-1,1] |
| T4 tied KV and sink | q=0, shared k=v=[2,4], sink=0, d=1, L=o | L=2; bar(logits)=[0,2/3], bar(sink)=-2/3, bar(q)=8/3, bar(shared latent)=[1/3,1/3] |
| T5 CE | logits=[0,0], target=0, m=1 | L=log(2), grad=[-1/2,1/2] |
| T6 valid-token reduction | scalar per-token derivatives: microbatch1=[2], microbatch2=[4,4,4] | D=4, grad=7/2; equal-microbatch average gives 3 and fails |
| T7 repeated gather | table=[3,7], IDs=[0,1,0], L=[1,2,4]·gather | L=29, table grad=[5,2]; integer IDs have no grad |
| T8 image repeat | x=2 repeated four times, L=[1,2,4,8]·repeat | L=30, bar(x)=15; detached encoder/dropped slot fails |
| T9 STE | x=0.3, supplied codec output=0.25, L=2*FQ(x) | L=0.5, bar(x)=2, scale grad absent; supplied value is not a codec byte fixture |
| T10 Sinkhorn two-step | W0=ones(2,2), G1=[[1,-1],[-1,1]], G2=0, eta=1, multiplier=1 | M1=.05*G1, N1=.0975*G1, M2=.0475*G1, N2=.045125*G1; Delta1≈Delta2≈G1, W2≈W0-.36*G1 within 1e-12 |
| T11 AdamW first step | W=2,g=4,m=v=0,eta=.01,lambda=.1 | m=.4,v=.8,mh=4,vh=16,Wnext≈1.988 within 1e-12; lambda=0 gives 1.99 |

T1–T8 permit finite differences with selections held fixed; T9 checks surrogate
backward directly; T10–T11 check next states/weights rather than differentiating
through the optimizer. Required extra implementation tests: CED norm chain,
nonuniform mHC predictor derivative, index-K/main-KV branching, nonzero-q tied
KV gradient, distinct owner contributions, actual B3 ranking margins, and A4
nonuniform S7. Passing these small cards alone is not complete B2-I acceptance.

## Active decisions and remaining unknowns

For this floating mathematical/CE/update-card specification active IDs are
O01,O02,O03,O04,O07,O10,O11,O13,O14,O16,O17. All are RESOLVED at pinned A8.
O05/O06 pretraining unfreeze schedules and O15 model FP4-off parity are inactive.
O08/O09 quantized table update/master representation and O12 indexer objective
remain OPEN. They are explicitly outside executable cards; enabling those paths
activates their gates and blocks acceptance. A scalar Sinkhorn card does not
choose a production Engram trainability mask or quantizer regeneration scheme.

Unresolved recipe inputs also include the trainability mask per phase, exact
sequence-balancing objective/activation, runtime loss masks/scaling ledger,
RL/OPD objectives, Muon backend approximation and native tolerances. Do not label
these as report facts or reopen resolved A8 choices. The pinned task matrix lists only O16 as the direct B2-S blocker; its
`blocked_by_open=true` metadata predates O16 resolution. The decision record
and successful active-O16 gate establish the resolved specification prerequisite.
B2-I still lists O12 and must not claim complete indexer training acceptance.

Run pinned `experimental/lite/docs/plans/validate_deepseek_v41_plan.py` separately
with each active `--active-decision ID`; run O08/O09/O12 as negative controls.
The validator verifies policy status/plan structure, not completeness of this
mathematics or justification for declaring a decision inactive. Review must check
that scope explicitly. No native/GPU, optimizer integration or full training
claim follows from this specification.
