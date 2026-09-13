"""Attribute only the current sparse-attention index backward nodes."""

import torch


def observed_attention(native):
    def forward(*args, **kwargs):
        output = native(*args, **kwargs)
        # Stop at this attention call's operands, excluding upstream layers.
        seen = {x.grad_fn for x in args[:3] if x.grad_fn is not None}

        def walk(node):
            if node is None or node in seen:
                return
            seen.add(node)
            if node.name() == 'IndexBackward0':
                active = []

                def begin(grads):
                    context = torch.profiler.record_function('QSA_KV_INDEX_BACKWARD')
                    context.__enter__()
                    active.append(context)

                def end(grad_inputs, grad_outputs):
                    active.pop().__exit__(None, None, None)

                node.register_prehook(begin)
                node.register_hook(end)
                return
            for child, _ in node.next_functions:
                walk(child)

        walk(output.grad_fn)
        return output

    return forward
