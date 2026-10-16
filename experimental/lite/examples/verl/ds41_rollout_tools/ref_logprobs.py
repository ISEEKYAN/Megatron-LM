#!/usr/bin/env python3
"""DS4.1 training-side reference logprobs (MLite, same source as the rollout).

Teacher-forces every (prompt, response) record of a rollout output through the
MLite training forward used by the DS4.1 proxy recipe (impl=lite, quantized W4A8 experts,
deployment_math, frozen Engram, full determinism, one sequence per forward as
ppo_micro_batch_size_per_gpu=1) and recomputes each response token's logprob
the way the actor does (logits / temperature -> deployment log_softmax ->
gather). Writes records in the rollout format (common.py), so compare.py can
diff rollout vs reference, or a modified rollout vs the frozen reference.

Launch with torchrun; ranks form the expert-parallel group (DP-only world shards
the Engram table as in production). Only rank 0 writes.
  torchrun --standalone --nproc_per_node 4 ref_logprobs.py --model M --records R --out O
See README.md for the runtime this was validated with.
"""
import argparse
import os


def parse():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--model', required=True, help='prefix model dir (same as the rollout)'
    )
    ap.add_argument(
        '--records',
        required=True,
        help='rollout output (.jsonl or .pt) whose tokens are scored',
    )
    ap.add_argument('--out', required=True, help='reference records (.jsonl or .pt)')
    ap.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='actor temperature (default %(default)s)',
    )
    ap.add_argument(
        '--ep',
        type=int,
        default=None,
        help='expert parallel size (default: WORLD_SIZE)',
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
        help='score only the first N records (rehearsal)',
    )
    return ap.parse_args()


def main():
    args = parse()
    os.environ.setdefault('VLLM_BATCH_INVARIANT', '1')
    import torch

    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world > 1:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        torch.distributed.init_process_group('nccl')
    try:  # same as actor.engine.full_determinism=true, seed=42
        from verl.workers.engine.utils import enable_full_determinism

        enable_full_determinism(seed=42)
    except ImportError:
        pass
    import json

    from common import load_records, save_records
    from megatron.lite.model.deepseek_v41.lite import protocol as P
    from megatron.lite.primitive.modules import deployment_math
    from megatron.lite.runtime.contracts import ParallelConfig

    token_map = json.load(open(os.path.join(args.model, 'token-map.json')))
    cfg = P.build_model_config(args.model)
    impl = P.ImplConfig(
        parallel=ParallelConfig(ep=args.ep or world),
        quantized=True,
        w4a8_experts=True,
        deployment_math=True,
        trainable_engram=False,
        use_deepep=False,
        dtype=torch.bfloat16,
        token_map=token_map,
    )
    bundle = P.build_model(cfg, impl_cfg=impl)
    model = bundle.chunks[0]
    P.load_hf_weights(model, args.model, cfg, bundle.parallel_state)
    model.eval()

    records = load_records(args.records)[: args.limit]
    out = []
    for r in records:
        prompt, response = r['prompt_ids'], r['response_ids']
        seq = torch.tensor(prompt + response, device='cuda')
        cu = torch.tensor([0, seq.numel()], dtype=torch.int32, device='cuda')
        with torch.no_grad():
            logits = model(seq[None], cu_seqlens=cu, return_head_hidden=False)['logits']
        logits = logits.reshape(seq.numel(), -1)[len(prompt) - 1 : -1]
        distribution = deployment_math.log_softmax(logits.float() / args.temperature)
        labels = torch.tensor(response, device='cuda')
        lp = distribution.gather(-1, labels[:, None]).squeeze(-1)
        out.append(
            {
                **r,
                'logprobs': [float(x) for x in lp.float().tolist()],
                'meta': dict(side='mlite', temperature=args.temperature, world=world),
            }
        )
    if rank == 0:
        save_records(out, args.out)
        print(json.dumps(dict(records=len(out), out=args.out)))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
