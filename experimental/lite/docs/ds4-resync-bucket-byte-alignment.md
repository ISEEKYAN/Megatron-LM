# DS4 resync IPC bucket byte-alignment defect (code-confirmed)

Scope: static source confirmation of why the 128-GPU real-weight (FP8) DeepSeek-V4
RL run crashes during the vLLM weight resync, while random-init BF16 proxy runs of
identical geometry stay green.

## Locus (version-dependent file layout)

The crash stack from the 128-GPU run named
`verl/workers/rollout/vllm_rollout/utils.py:231` (`buffer[offset:offset+size].view(...)`).
In the current `verl_flameagainst` tree that receive path has been refactored out of
`utils.py` and now lives in the bucketed IPC transfer helper below. Same defect, different
file layout across verl versions — confirm which verl the run actually used (container
site-packages vs `verl_flameagainst`) before landing a patch.

- Sender: `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`
  - line 151: `self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), ...)`
  - line 152: `offset += weight.nbytes`  ← no alignment padding
  - line 148: the per-tensor `offset` is recorded into `bucket_meta[name]["offset"]`
- Receiver: same file
  - line 282: `size = dtype.itemsize * shape.numel()`
  - line 283: `tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)`

The receiver reads `offset` straight from the sender-provided `bucket_meta`, so sender
and receiver share one offset value. The buffer itself is `torch.uint8`.

## Mechanism

`torch.Tensor.view(dtype=D)` on a uint8 slice requires the slice's byte
`storage_offset` to be divisible by `D.itemsize`. The sender advances `offset` by
`weight.nbytes = numel * dtype.itemsize`, with no realignment between tensors.

- Pure BF16/FP32 stream (proxy): every `nbytes` is a multiple of 2 (BF16) or 4 (FP32),
  so `offset` stays aligned to any subsequent tensor's itemsize. The receive-side
  `.view(dtype)` never trips. → proxy is always green.
- Real FP8 weights (itemsize = 1): an FP8 tensor with an **odd** byte count leaves
  `offset` at an odd value. The next non-FP8 (BF16/FP32) tensor in the same bucket then
  hits `.view(dtype=...)` at an unaligned `storage_offset` → `RuntimeError: storage_offset
  must be divisible by <itemsize>`.

This asymmetry (proxy always green, real FP8 crashes) is fully explained by the code and
is independent of OOM / offload / gpu-memory-utilization. It is a distinct defect from the
separate export-side FP8 quantization concern (`quantize_block_fp8._validate` on non-128
divisible shapes), which fires earlier, on the trainer/export side, before IPC.

## Two candidate crash sites (to be disambiguated by the isolation reproducer)

1. Export side — `quantize_block_fp8._validate` ValueError (namespace/shape). Visible
   export-only, before any vLLM/IPC. 
2. Receive side — `bucketed_weight_transfer.py:283` `.view()` storage_offset alignment.
   Requires the full IPC send→receive path (i.e. a live vLLM receiver).

An export-only run distinguishes them: if export raises, it is (1); if export completes
cleanly, the crash is (2), the byte-alignment defect documented here.

## Fix options

- A — sender-side alignment (root fix, upstream verl): pad `offset` up to an 8-byte
  boundary before recording it and before the copy, e.g. `offset = (offset + 7) & ~7`.
  Because `bucket_meta` carries the padded offset, the receiver needs no change and stays
  byte-consistent for any dtype up to 8 bytes. ~2 lines, but lands in upstream verl.
- B — receiver-side clone (upstream verl): `self.buffer[offset:offset+size].clone().view(dtype)`.
  The clone yields storage_offset 0, satisfying the view constraint, at the cost of one
  extra copy per tensor.
- C — verl_mlite wrapper monkeypatch (in-repo, `experimental/lite/examples/verl/verl_mlite/`):
  wrap the sender/receiver without touching upstream verl. Lands in a repo we control.

Fix A is the cleanest root fix; fix C is the only option that stays inside this repo's
write scope.
