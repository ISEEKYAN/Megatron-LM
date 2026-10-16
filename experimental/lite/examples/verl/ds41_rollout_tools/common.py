"""Shared record format for the DS4.1 rollout/reference logprob tools.

One record per (prompt, sample):
  {"id": str, "sample": int, "prompt_ids": [int], "response_ids": [int],
   "logprobs": [float], "meta": {...}}
``logprobs[i]`` is the log-probability of ``response_ids[i]``. Values are FP32
widened to Python float (exact), so JSONL round-trips bit for bit; a .pt file
holds the same records with FP32 tensors.
"""

import hashlib
import json
from pathlib import Path


def ids_sha256(ids):
    import torch

    return hashlib.sha256(
        torch.tensor(ids, dtype=torch.int64).numpy().tobytes()
    ).hexdigest()


def load_prompts(path):
    """Prompt set: a .pt dict {"prompts": [{"id", "prompt_ids"}], ...} or JSONL."""
    path = Path(path)
    if path.suffix == '.pt':
        import torch

        data = torch.load(path, weights_only=True)
        prompts = data['prompts']
    else:
        prompts = [json.loads(line) for line in path.read_text().splitlines() if line]
    for p in prompts:
        p['prompt_ids'] = [int(t) for t in p['prompt_ids']]
        if 'sha256' in p and p['sha256'] != ids_sha256(p['prompt_ids']):
            raise ValueError(f"prompt {p['id']} does not match its sha256")
    return prompts


def save_records(records, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.pt':
        import torch

        torch.save(
            [
                {**r, 'logprobs': torch.tensor(r['logprobs'], dtype=torch.float32)}
                for r in records
            ],
            out,
        )
    else:
        with out.open('w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')


def load_records(path):
    path = Path(path)
    if path.suffix == '.pt':
        import torch

        recs = torch.load(path, weights_only=False)
        for r in recs:
            r['logprobs'] = [float(x) for x in r['logprobs'].float().tolist()]
        return recs
    return [json.loads(line) for line in path.read_text().splitlines() if line]
