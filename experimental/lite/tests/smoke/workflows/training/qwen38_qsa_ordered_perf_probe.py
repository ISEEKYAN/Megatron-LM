"""CP2-only cost prototype; no product/reference policy changes."""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.lite.model.qwen3_8_flash_next.math import sparse_attention
from torch.distributed.elastic.multiprocessing.errors import record


def selected_adjoints(record, left, right):
    inputs = [record[f'input{i}'].cuda().clone().requires_grad_() for i in range(3)]
    q, k, v = inputs
    routes = record['input3'][:, left:right].cuda()
    output = sparse_attention(q[:, left:right], k, v, routes)
    captured, handles, seen = {}, [], set()

    def walk(node):
        if node is None or node in seen:
            return
        seen.add(node)
        if node.name() == 'IndexBackward0':
            parent = node.next_functions[0][0]
            variable = getattr(parent, 'variable', None)
            key = 'k' if variable is k else 'v' if variable is v else None
            assert key is not None, 'QSA_INDEX_LEAF_IDENTITY'

            def save(grads, key=key):
                captured[key] = grads[0].detach().clone()

            handles.append(node.register_prehook(save))
        for child, _ in node.next_functions:
            walk(child)

    walk(output.grad_fn)
    output.backward(record['output_grad'][:, left:right].cuda())
    for handle in handles:
        handle.remove()
    assert set(captured) == {'k', 'v'}, 'QSA_INDEX_ADJOINT_CAPTURE_REQUIRED'
    return dict(routes=routes, selected=captured, dk=k.grad, dv=v.grad)


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calls', type=int, default=1024)
    parser.add_argument('--rounds', type=int, default=21)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE']) == 2, 'QSA_COST_CP2_ONLY'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    args.output.mkdir(parents=True, exist_ok=True)
    serial = torch.load(args.source / 'serial.pt', weights_only=True)['modules']['QSA']
    actual = torch.load(args.source / f'cp2/rank{rank}.pt', weights_only=True)[
        'modules'
    ]['QSA']
    documents = [r for r in serial['boundaries'] if r['kind'] == 'attention']
    boundaries = [0, 5, 13, 16]
    full_records, local_records, checks = [], [], []

    def exact(name, a, b):
        byte_equal = torch.equal(
            a.detach().cpu().contiguous().view(torch.uint8),
            b.detach().cpu().contiguous().view(torch.uint8),
        )
        checks.append(
            dict(
                name=name,
                byte_equal=byte_equal,
                different=int((a.detach().cpu() != b.detach().cpu()).sum()),
            )
        )
        assert byte_equal, (name, rank, checks[-1])

    for doc, (start, end) in zip(documents, zip(boundaries, boundaries[1:])):
        complete = selected_adjoints(doc, 0, end - start)
        for key, original in [('dk', 'input1_grad'), ('dv', 'input2_grad')]:
            exact('QSA_NATIVE_FULL_SERIAL_' + key, complete[key], doc[original])
        full_records.append(complete)
        left, right = max(start, rank * 8), min(end, rank * 8 + 8)
        if left >= right:
            continue
        local = selected_adjoints(doc, left - start, right - start)
        for key in ('k', 'v'):
            exact(
                'QSA_SELECTED_CONTRIBUTION_SERIAL_' + key,
                local['selected'][key],
                complete['selected'][key][:, left - start : right - start],
            )
        local.update(start=start, end=end)
        # Retain the actual native IndexBackward graph for the baseline.
        local['leaves'], local['indexed'] = {}, {}
        batch = torch.arange(1, device='cuda')[:, None, None]
        local['indices'] = [batch, local['routes'].clamp_min(0)]
        for key in ('k', 'v'):
            leaf = torch.zeros_like(doc['input1']).cuda().requires_grad_()
            local['leaves'][key] = leaf
            local['indexed'][key] = leaf[batch, local['indices'][1]]
        local_records.append(local)
    reference = torch.stack(
        [torch.cat([r[key] for r in full_records], 1) for key in ('dk', 'dv')]
    )
    carry = torch.zeros_like(reference)
    partial = torch.zeros_like(reference)
    output = torch.empty_like(reference[:, :, :8]).contiguous()
    send_back = torch.empty_like(output)

    def baseline():
        partial.zero_()
        for record in local_records:
            for slot, key in enumerate(('k', 'v')):
                (gradient,) = torch.autograd.grad(
                    record['indexed'][key],
                    record['leaves'][key],
                    record['selected'][key],
                    retain_graph=True,
                )
                partial[slot, :, record['start'] : record['end']].copy_(gradient)
        for slot in range(2):
            dist.reduce_scatter(output[slot], list(partial[slot].chunk(2, dim=1)))

    def ordered():
        if rank == 0:
            carry.zero_()
        else:
            dist.recv(carry, src=0)
        for record in local_records:
            for slot, key in enumerate(('k', 'v')):
                torch.ops.aten._index_put_impl_(
                    carry[slot, :, record['start'] : record['end']],
                    record['indices'],
                    record['selected'][key],
                    True,
                    True,
                )
        if rank == 0:
            dist.send(carry, dst=1)
            dist.recv(output, src=1)
        else:
            output.copy_(carry[:, :, 8:])
            send_back.copy_(carry[:, :, :8])
            dist.send(send_back, dst=0)

    raw = dict(
        reference=reference.cpu(),
        local=[
            dict(
                start=r['start'],
                end=r['end'],
                routes=r['routes'].cpu(),
                selected={k: v.cpu() for k, v in r['selected'].items()},
            )
            for r in local_records
        ],
    )
    report = dict(
        rank=rank,
        calls=args.calls,
        rounds=args.rounds,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        checks=checks,
        samples_us={'partial': [], 'ordered': []},
    )
    graphs = {}
    try:
        baseline()
        raw['partial'] = output.cpu().clone()
        observed = [r for r in actual['boundaries'] if r['kind'] == 'gather']
        exact('QSA_NATIVE_PARTIAL_K', output[0], observed[1]['input0_grad'])
        exact('QSA_NATIVE_PARTIAL_V', output[1], observed[2]['input0_grad'])
        ordered()
        raw['ordered'] = output.cpu().clone()
        exact(
            'QSA_ORDERED_TRUE_SERIAL_BITWISE',
            output,
            reference[:, :, rank * 8 : rank * 8 + 8],
        )
        for _ in range(3):
            baseline()
            exact('QSA_PARTIAL_REPEAT_BITWISE', output, raw['partial'])
            ordered()
            exact('QSA_ORDERED_REPEAT_BITWISE', output, raw['ordered'])
        print('QSA_ORDERED_CORRECTNESS', rank, json.dumps(checks), flush=True)
        graphs = {}
        # Autograd schedules a backward node on its forward stream. Build the
        # retained IndexBackward graphs on the same non-default capture stream.
        graph_stream = torch.cuda.Stream()
        graph_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(graph_stream):
            for record in local_records:
                for key in ('k', 'v'):
                    leaf = record['leaves'][key].detach().clone().requires_grad_()
                    record['leaves'][key] = leaf
                    record['indexed'][key] = leaf[
                        record['indices'][0], record['indices'][1]
                    ]
        torch.cuda.synchronize()
        for name, function in [('partial', baseline), ('ordered', ordered)]:
            with torch.cuda.stream(graph_stream):
                for _ in range(64):
                    function()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_stream):
                for _ in range(args.calls):
                    function()
            graphs[name] = graph
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            exact('QSA_GRAPH_' + name.upper() + '_BITWISE', output, raw[name])
            print('QSA_COST_GRAPH_READY', rank, name, flush=True)
        for iteration in range(args.rounds):
            for name in (
                ('partial', 'ordered') if iteration % 2 == 0 else ('ordered', 'partial')
            ):
                dist.barrier()
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                begin.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                report['samples_us'][name].append(
                    begin.elapsed_time(end) * 1000 / args.calls
                )
        ratio = [
            x / y
            for x, y in zip(
                report['samples_us']['ordered'], report['samples_us']['partial']
            )
        ]
        report['partial_us'] = statistics.median(report['samples_us']['partial'])
        report['ordered_us'] = statistics.median(report['samples_us']['ordered'])
        report['paired_ratio'] = statistics.median(ratio)
        report['paired_ratio_p10_p90'] = (
            torch.tensor(ratio, dtype=torch.float64)
            .quantile(torch.tensor([0.1, 0.9], dtype=torch.float64))
            .tolist()
        )
        print('QSA_ORDERED_COST', rank, json.dumps(report), flush=True)
    finally:
        torch.save(raw, args.output / f'raw-rank{rank}.pt')
        (args.output / f'report-rank{rank}.json').write_text(
            json.dumps(report, indent=2) + '\n'
        )
        print('QSA_COST_RELEASE_GRAPHS_BEGIN', rank, flush=True)
        for graph in graphs.values():
            graph.reset()
        graphs.clear()
        torch.cuda.synchronize()
        print('QSA_COST_RELEASE_GRAPHS_DONE', rank, flush=True)
        dist.destroy_process_group()
        print('QSA_COST_GROUP_DESTROYED', rank, flush=True)


if __name__ == '__main__':
    main()
