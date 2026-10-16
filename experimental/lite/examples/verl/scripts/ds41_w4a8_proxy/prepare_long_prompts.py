"""Prepare long-prompt coverage, with explicit synthetic repetition if requested."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-tokens', type=int, default=1280)
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--repeat-to-limit', action='store_true')
    parser.add_argument('--repeat-content-to-min', action='store_true')
    args = parser.parse_args()
    if args.min_tokens <= 1024 or args.limit < 16:
        parser.error('Use >1024 prompt tokens and at least sixteen prompts')
    config = json.loads((args.model / 'config.json').read_text())['text_config']
    if config['num_hidden_layers'] != 4:
        parser.error('The index audit requires the four-layer prefix')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    data = pd.read_parquet(args.data)
    template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
    selected, lengths, originals, repetitions, prepared = [], [], [], [], []
    for index, prompt in enumerate(data['prompt']):
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        tokens = tokenizer.apply_chat_template(
            prompt, chat_template=template, tokenize=True, return_dict=False
        )
        original_length = len(tokens)
        qualifying_length = 1280 if args.repeat_content_to_min else args.min_tokens
        if original_length >= qualifying_length:
            copies = 1
            if args.repeat_content_to_min:
                if len(prompt) != 1 or prompt[0]['role'] != 'user':
                    raise ValueError('Synthetic stress inputs require one user message')
                content = prompt[0]['content']
                if not isinstance(content, str):
                    raise ValueError(
                        'Synthetic stress input content must be plain text'
                    )
                while len(tokens) < args.min_tokens:
                    copies += 1
                    prompt = [dict(prompt[0], content='\n\n'.join([content] * copies))]
                    tokens = tokenizer.apply_chat_template(
                        prompt, chat_template=template, tokenize=True, return_dict=False
                    )
            selected.append(index)
            lengths.append(len(tokens))
            originals.append(original_length)
            repetitions.append(copies)
            prepared.append(prompt)
            if len(selected) == args.limit:
                break
    if selected and len(selected) < args.limit and args.repeat_to_limit:
        copies = (args.limit + len(selected) - 1) // len(selected)
        selected = (selected * copies)[: args.limit]
        lengths = (lengths * copies)[: args.limit]
        originals = (originals * copies)[: args.limit]
        repetitions = (repetitions * copies)[: args.limit]
        prepared = (prepared * copies)[: args.limit]
    if len(selected) != args.limit:
        raise RuntimeError(
            f'Only {len(selected)} sufficiently long prompts in the input'
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = data.iloc[selected].copy()
    if args.repeat_content_to_min:
        output['prompt'] = prepared
    output.to_parquet(args.output, index=False)
    receipt = {
        'source_sha256': hashlib.file_digest(
            args.data.open('rb'), 'sha256'
        ).hexdigest(),
        'output_sha256': hashlib.file_digest(
            args.output.open('rb'), 'sha256'
        ).hexdigest(),
        'source_row_indices': selected,
        'unique_source_rows': len(set(selected)),
        'repeated_to_limit': len(set(selected)) < len(selected),
        'prompt_lengths_before_truncation': lengths,
        'original_prompt_lengths': originals,
        'content_repetitions': repetitions,
        'synthetic_stress_input': args.repeat_content_to_min,
        'min_prompt_tokens': args.min_tokens,
        'chat_template': template,
        'selection': (
            'synthetic stress inputs: repeated original user text with blank-line separators; not independent questions or a quality evaluation'
            if args.repeat_content_to_min
            else 'first qualifying rows in source order; prompt contents unchanged'
        ),
    }
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('DS41_LONG_PROMPTS_PREPARED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
