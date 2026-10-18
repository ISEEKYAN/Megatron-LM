"""Exercise actual zero-GPU Ray workers with the recipe audit/import environment."""

import json
import os
import sys

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

assert ray.is_initialized() and ray.cluster_resources().get('GPU', 0) == 0
recipe = '/opt/src/mlite/experimental/lite/examples/verl/scripts/ds41_w4a8_proxy'
environment = dict(
    VERL_MLITE_SKIP_RUNTIME_PATCHES='0',
    VERL_MLITE_HF_CONFIG_MODEL_TYPE='deepseek_v41',
    VLLM_BATCH_INVARIANT='1',
    W4_GRPO_AUDIT='1',
    W4_GRPO_OUT='/test-output/worker-optimizer-probe',
    DS41_INDEXER_DIAGNOSTIC=str(int(os.environ.get('RECIPE_C8_MODE') == '--c8-4l')),
    PYTHONPATH=recipe + '/audit:' + recipe + ':' + os.environ['PYTHONPATH'],
)


@ray.remote(num_cpus=1)
def check_optimizer_worker():
    import os
    import socket

    import emerging_optimizers.orthogonalized_optimizers.muon_utils as utils
    import torch
    from megatron.lite.primitive.optimizers.headwise_muon import HeadwiseMuon

    assert os.environ['RAY_AUTH_MODE'] == 'token' and os.environ['RAY_AUTH_TOKEN']
    assert os.environ['VERL_MLITE_SKIP_RUNTIME_PATCHES'] == '0'
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    assert utils.__file__.startswith('/opt/src/optimizer-deps/')
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    torch.backends.cudnn.rnn.fp32_precision = 'ieee'
    parameter = torch.nn.Parameter(torch.eye(4))
    before = parameter.detach().clone()
    optimizer = HeadwiseMuon(
        [dict(params=[parameter], matrix_shape=(4, 4))],
        lr=0.001,
        ns_steps=2,
        coefficient_type='quintic',
    )
    parameter.grad = torch.ones_like(parameter)
    assert optimizer.prepare_step()
    optimizer.commit_step()
    assert not torch.equal(parameter, before) and torch.isfinite(parameter).all()
    return dict(
        host=socket.gethostname(),
        pid=os.getpid(),
        cuda_initialized=False,
        actual_muon_update=True,
        import_path=utils.__file__,
    )


nodes = [node['NodeID'] for node in ray.nodes() if node['Alive']]
assert len(nodes) == int(os.environ.get('RECIPE_NNODES', '2'))
futures = [
    check_optimizer_worker.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
        runtime_env=dict(env_vars=environment, py_executable=sys.executable),
    ).remote()
    for node in nodes
]
results = ray.get(futures, timeout=240)
assert len({row['host'] for row in results}) == int(
    os.environ.get('RECIPE_NNODES', '2')
)
print('RECIPE_ACTUAL_CPU_OPTIMIZER_WORKERS_OK', json.dumps(results), flush=True)
