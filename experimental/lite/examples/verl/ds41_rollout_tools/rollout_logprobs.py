#!/usr/bin/env python3
"""DS4.1 rollout-side logprobs: vLLM only (no Ray, no verl).

Loads a DS4.1 prefix model dir (4L, or the 40-layer model) with the rollout
engine config of the DS4.1 proxy recipe (batch invariance,
FLASHMLA_SPARSE_DSV41, NVFP4 KV, MXFP4 index K, DeepGEMM MoE,
allgather_reducescatter, no prefix cache, max_num_batched_tokens =
max_model_len, CUDA graph FULL_AND_PIECEWISE capped at 16) and samples every
prompt n times, writing the logprob of each generated token (raw logprobs) as
JSONL or .pt (format: common.py).

Parallelism: TP1, DP = EP = nnodes x gpus-per-node, vLLM's native offline
data parallel (one engine process per GPU; no Ray). Run the same command on
every node with its --node-rank; node 0's address is --master-addr. Each DP
rank takes a fixed round-robin slice of the (prompt, sample) requests and
submits it in one generate call so all ranks step together. Each rank writes
<out>.rank<k>.jsonl on the shared filesystem; node 0 merges them in prompt
order into <out>.

Examples
  4L, one GPU:   rollout_logprobs.py --model M4 --prompts P.pt --out r.jsonl
  40L, 2x4 GPUs: rollout_logprobs.py --model M40 --prompts P.pt --out r.jsonl \\
                   --nnodes 2 --gpus-per-node 4 --node-rank $SLURM_NODEID \\
                   --master-addr <node0> --master-port 29551
See README.md for the runtime this was validated with.
"""
import argparse
import os
import sys
import time


def parse(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--model',
        required=True,
        help='prefix model dir (config.json, weights, token-map.json)',
    )
    ap.add_argument(
        '--prompts',
        required=True,
        help='prompt set from make_prompts.py (.pt or .jsonl)',
    )
    ap.add_argument(
        '--out', required=True, help='merged output records (.jsonl or .pt)'
    )
    ap.add_argument(
        '--response-length',
        type=int,
        default=128,
        help='max new tokens, ignore_eos (default %(default)s)',
    )
    ap.add_argument(
        '--n',
        type=int,
        default=4,
        help='samples per prompt (default %(default)s, as the DS4.1 proxy recipe)',
    )
    ap.add_argument(
        '--seed',
        type=int,
        default=42,
        help='sampling seed base; request seed = seed + 1000*prompt + sample (default %(default)s)',
    )
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument(
        '--cudagraph',
        choices=('full_and_piecewise', 'none'),
        default='full_and_piecewise',
        help='the release recipe uses full_and_piecewise; none = enforce_eager (default %(default)s)',
    )
    ap.add_argument(
        '--max-capture',
        type=int,
        default=16,
        help='max_cudagraph_capture_size (default %(default)s)',
    )
    ap.add_argument(
        '--precision',
        choices=('official', 'release'),
        default='official',
        help='official requires text_config expert_activation_group_size=32, wo_a_dtype=bf16 (default %(default)s)',
    )
    ap.add_argument(
        '--nnodes',
        type=int,
        default=1,
        help='nodes in the DP group (default %(default)s)',
    )
    ap.add_argument(
        '--gpus-per-node',
        type=int,
        default=1,
        help='DP ranks (GPUs) per node (default %(default)s)',
    )
    ap.add_argument(
        '--node-rank',
        type=int,
        default=0,
        help='this node, 0..nnodes-1 (default %(default)s)',
    )
    ap.add_argument(
        '--master-addr',
        default='127.0.0.1',
        help='node 0 address for DP coordination (default %(default)s)',
    )
    ap.add_argument(
        '--master-port',
        type=int,
        default=29551,
        help='DP coordination port on node 0 (default %(default)s)',
    )
    ap.add_argument(
        '--max-num-seqs',
        type=int,
        default=4,
        help='(default %(default)s, as the DS4.1 proxy recipe)',
    )
    ap.add_argument(
        '--kv-cache-bytes',
        type=int,
        default=None,
        help='kv_cache_memory_bytes (default: from --gpu-memory-utilization)',
    )
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.92)
    ap.add_argument(
        '--merge-timeout',
        type=int,
        default=1800,
        help='seconds node 0 waits for other ranks\' parts (default %(default)s)',
    )
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='validate config and print the per-rank plan and engine args; no GPU',
    )
    ap.add_argument('--local-rank', type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not 0 <= args.node_rank < args.nnodes or args.gpus_per_node < 1:
        ap.error('need 0 <= --node-rank < --nnodes and --gpus-per-node >= 1')
    if args.nnodes > 1 and args.master_addr in ('127.0.0.1', 'localhost'):
        ap.error('multi-node runs need --master-addr set to node 0\'s address')
    return args


def dp_size(args):
    return args.nnodes * args.gpus_per_node


def check_precision(model, precision):
    import json

    text = json.load(open(os.path.join(model, 'config.json')))['text_config']
    official = (text.get('expert_activation_group_size'), text.get('wo_a_dtype')) == (
        32,
        'bf16',
    )
    if official != (precision == 'official'):
        sys.exit(
            f'--precision {precision} does not match {model}/config.json text_config'
        )
    return text


def requests(args, prompts):
    """All (prompt index, sample) requests in a fixed global order."""
    return [(i, s) for i in range(len(prompts)) for s in range(args.n)]


def rank_requests(args, prompts, rank):
    return [
        r for k, r in enumerate(requests(args, prompts)) if k % dp_size(args) == rank
    ]


def engine_args(args, prompts):
    max_len = max(len(p['prompt_ids']) for p in prompts) + args.response_length
    graph = args.cudagraph == 'full_and_piecewise'
    kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        dtype='bfloat16',
        seed=args.seed,
        tensor_parallel_size=1,
        kv_cache_dtype='nvfp4_ds_mla',
        attention_backend='FLASHMLA_SPARSE_DSV41',
        attention_config={'indexer_kv_dtype': 'mxfp4', 'indexer_sparse_logits': False},
        hf_overrides={'head_dtype': 'float32'},
        moe_backend='deep_gemm',
        compilation_config=(
            {
                'mode': 0,
                'cudagraph_mode': 'FULL_AND_PIECEWISE',
                'max_cudagraph_capture_size': args.max_capture,
            }
            if graph
            else {'mode': 0, 'cudagraph_mode': 'NONE'}
        ),
        enforce_eager=not graph,
        engram_config={
            'cpu_offload': False,
            'embedding_across_dp': True,
            'dp_shared_memory': False,
        },
        enable_prefix_caching=False,
        max_model_len=max_len,
        max_num_batched_tokens=max_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={'image': 0},
    )
    if args.kv_cache_bytes:
        kwargs['kv_cache_memory_bytes'] = args.kv_cache_bytes
    if dp_size(args) > 1:
        kwargs.update(
            enable_expert_parallel=True, all2all_backend='allgather_reducescatter'
        )
    return kwargs


def rank_env(args, local_rank):
    env = {'VLLM_BATCH_INVARIANT': '1'}
    if args.cudagraph == 'full_and_piecewise':
        env['VLLM_USE_BREAKABLE_CUDAGRAPH'] = '1'
    if dp_size(args) > 1:
        env.update(
            VLLM_DP_RANK=str(args.node_rank * args.gpus_per_node + local_rank),
            VLLM_DP_RANK_LOCAL=str(local_rank),
            VLLM_DP_SIZE=str(dp_size(args)),
            VLLM_DP_MASTER_IP=args.master_addr,
            VLLM_DP_MASTER_PORT=str(args.master_port),
        )
    return env


def part_path(args, rank):
    return f'{args.out}.rank{rank}.jsonl'


def rank_main(args, prompts):
    import json

    from common import save_records
    from vllm import LLM, SamplingParams

    rank = args.node_rank * args.gpus_per_node + args.local_rank
    mine = rank_requests(args, prompts, rank)
    llm = LLM(**engine_args(args, prompts))
    params = [
        SamplingParams(
            max_tokens=args.response_length,
            temperature=args.temperature,
            top_p=1.0,
            top_k=-1,
            ignore_eos=True,
            seed=args.seed + 1000 * i + s,
            logprobs=0,
        )
        for i, s in mine
    ]
    # An idle rank still steps with the others (placeholder request, discarded).
    inputs = [{'prompt_token_ids': prompts[i]['prompt_ids']} for i, _ in mine] or [
        {'prompt_token_ids': [0]}
    ]
    params = params or [SamplingParams(max_tokens=1)]
    outs = llm.generate(inputs, params, use_tqdm=False)
    records = []
    for (i, s), out in zip(mine, outs):
        o = out.outputs[0]
        if any(lp is None for lp in o.logprobs):
            raise RuntimeError('vLLM returned no logprob for a token')
        records.append(
            dict(
                id=prompts[i]['id'],
                sample=s,
                index=i,
                prompt_ids=prompts[i]['prompt_ids'],
                response_ids=list(o.token_ids),
                logprobs=[
                    float(lp[t].logprob) for t, lp in zip(o.token_ids, o.logprobs)
                ],
                meta=dict(
                    side='vllm',
                    cudagraph=args.cudagraph,
                    seed=args.seed + 1000 * i + s,
                    dp_rank=rank,
                    dp_size=dp_size(args),
                ),
            )
        )
    save_records(records, part_path(args, rank))
    print(json.dumps(dict(rank=rank, records=len(records), out=part_path(args, rank))))
    time.sleep(
        1
    )  # let the engines leave their loops together, as vLLM's DP example does


def merge(args, prompts):
    from common import load_records, save_records

    parts = [part_path(args, r) for r in range(dp_size(args))]
    deadline = time.time() + args.merge_timeout
    while not all(os.path.exists(p) for p in parts):
        if time.time() > deadline:
            sys.exit(
                f'missing rank outputs: {[p for p in parts if not os.path.exists(p)]}'
            )
        time.sleep(5)
    records = sorted(
        (r for p in parts for r in load_records(p)),
        key=lambda r: (r['index'], r['sample']),
    )
    if [(r['index'], r['sample']) for r in records] != requests(args, prompts):
        sys.exit('merged records do not cover every (prompt, sample) exactly once')
    save_records(records, args.out)
    print(f'merged {len(records)} records -> {args.out}')


def main():
    args = parse()
    from common import load_prompts

    text = check_precision(args.model, args.precision)
    prompts = load_prompts(args.prompts)
    if args.local_rank is not None:  # engine process for one local GPU
        os.environ.update(rank_env(args, args.local_rank))
        return rank_main(args, prompts)
    if args.dry_run:
        import json

        plan = dict(
            model_layers=text['num_hidden_layers'],
            dp_size=dp_size(args),
            node_rank=args.node_rank,
            ranks={
                args.node_rank * args.gpus_per_node
                + k: dict(
                    env=rank_env(args, k),
                    requests=len(
                        rank_requests(
                            args, prompts, args.node_rank * args.gpus_per_node + k
                        )
                    ),
                )
                for k in range(args.gpus_per_node)
            },
            engine_args=engine_args(args, prompts),
        )
        print(json.dumps(plan, indent=1, default=str))
        return
    import subprocess

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                *sys.argv[1:],
                '--local-rank',
                str(k),
            ]
        )
        for k in range(args.gpus_per_node)
    ]
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f'engine processes failed: {codes}')
    if args.node_rank == 0:
        merge(args, prompts)


if __name__ == '__main__':
    main()
