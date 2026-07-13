# Draft PR comment (English) — for maintainer review before posting

> Draft only. Do not post until reviewed/approved.

---

Thanks for tackling the long-sequence OOM here — the online-softmax + backward
recompute is correct and the parity tests are appreciated.

Before we take this, I want to flag that it works around a symptom rather than the
root cause. The CSA reference this module was ported from
(`megatron/core/transformer/experimental_attention_variant/csa.py`,
`unfused_compressed_sparse_attn`) is **sparse by construction**: it gathers only
the sliding-window ∪ compressed top-k keys per query, so the score tensor is
`[b, np, sq, topk]` with `topk = window + index_topk` — a fixed constant,
independent of sequence length. Our fallback here instead materializes the full
dense `[B, H, Sq, Skv]` score matrix and applies the window/top-k as `-inf` masks
(the indexer top-k at is used only to build a mask, never to shrink the tensor).
That dense materialization is the OOM source, and CP only relieves it linearly
(`O(S²/cp)`), which is why long context still blows up.

This PR keeps that dense algorithm and bounds *peak* memory by chunking the query
dim, but compute stays `O(S²)` and it adds a large hand-rolled autograd surface to
reproduce what the reference gets from a ~60-line sparse gather. We'd prefer to
fix the port to match the reference — gather only the top-k keys (the top-k
indices are already computed here) and score `[B,H,Sq,topk]` — which removes the
OOM at the root and keeps CP memory sequence-length-independent.

We think the streaming-softmax technique still has value in the narrow cases where
a genuinely dense score matrix is unavoidable (e.g. a dense-mode / explicit-mask
path, or 128×-compress "attend-to-all" layers where `topk ≈ full`), and would be
happy to keep it as an opt-in there.

Minor: `torch.equal` on GPU tensors in the forward introduces a device-host sync;
worth dropping or guarding behind a debug flag (as the review bot noted).
