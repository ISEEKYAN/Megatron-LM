#!/usr/bin/env python3
"""Build the fixed prompt set (token ids + sha256) from the DAPO parquet.

Always includes the 12800-token prompt that crosses the 12288 indexer
threshold, plus deterministic short and mid-length
prompts: the first rows of the parquet whose chat-templated length falls in
each requested bucket.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokenizer', required=True, help='model dir with tokenizer files')
    ap.add_argument('--data', required=True, help='DAPO training parquet')
    ap.add_argument(
        '--long-prompt',
        required=True,
        help='the 12800-token prompt: a .pt with prompt_ids, or a prompts .jsonl holding '
        'the long-12800 record (as shipped in prompts/ds41-v14-prompts.jsonl)',
    )
    ap.add_argument(
        '--buckets',
        default='32:128,128:512,512:2048,2048:8192',
        help='comma list of [lo:hi) token-length buckets, one prompt each (default %(default)s)',
    )
    ap.add_argument('--per-bucket', type=int, default=1)
    ap.add_argument(
        '--out', required=True, help='output .pt (also writes .jsonl next to it)'
    )
    args = ap.parse_args()

    import json
    from pathlib import Path

    import pandas as pd
    import torch
    from common import ids_sha256
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.long_prompt.endswith('.jsonl'):
        long_ids = next(
            json.loads(line)['prompt_ids']
            for line in Path(args.long_prompt).read_text().splitlines()
            if line and json.loads(line)['id'] == 'long-12800'
        )
    else:
        long_ids = torch.load(args.long_prompt, weights_only=True)[
            'prompt_ids'
        ].tolist()
    prompts = [dict(id='long-12800', prompt_ids=long_ids)]
    frame = pd.read_parquet(args.data)
    buckets = [tuple(map(int, b.split(':'))) for b in args.buckets.split(',')]
    found = {b: 0 for b in buckets}
    # The recipe's prompt rendering (prepare_long_prompts.py): BOS, then the
    # message contents concatenated, no role markers.
    template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
    for row, item in enumerate(frame['prompt']):
        item = item.tolist() if hasattr(item, 'tolist') else list(item)
        ids = tok.apply_chat_template(
            item, chat_template=template, tokenize=True, return_dict=False
        )
        for lo, hi in buckets:
            if lo <= len(ids) < hi and found[(lo, hi)] < args.per_bucket:
                found[(lo, hi)] += 1
                prompts.append(dict(id=f'row{row}-{len(ids)}', prompt_ids=list(ids)))
        if all(v == args.per_bucket for v in found.values()):
            break
    for p in prompts:
        p['sha256'] = ids_sha256(p['prompt_ids'])
    out = Path(args.out)
    torch.save({'prompts': prompts, 'source': vars(args)}, out)
    out.with_suffix('.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in prompts))
    print(json.dumps([(p['id'], len(p['prompt_ids'])) for p in prompts]))


if __name__ == '__main__':
    main()
