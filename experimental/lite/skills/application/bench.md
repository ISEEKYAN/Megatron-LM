# Bench Skill

Run benchmark-style checks without weakening precision evidence.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "application.bench", kind="state_machine", purpose="benchmark with controlled variables and evidence",
    imports=["basic.constitution"], calls=["perf.measure"],
    inputs=["task", "bench_config", "target", "budget"],
    outputs=["bench", "evidence", "risks"], exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def bench(task, bench_config, target, budget):
    controlled = bench_config.variables.freeze_all_except(bench_config.axis)
    if controlled.has_unknown_unfrozen_axes():
        return blocked("bench has uncontrolled variables", risks=controlled.unknown_axes)

    if bench_config.axis == "chunked_ep":
        target = require_public_bench_target(
            target,
            path="experimental/lite/examples/bench/bench.py",
            config_transport="--impl-cfg-json",
            required_fields=[
                "use_deepep", "enable_ep_chunk_overlap",
                "ep_chunk_max_token_rows_per_rank", "ep_chunk_full_recompute",
            ],
        )
        if not target.done:
            return blocked("public bench does not consume ChunkedEP config", evidence=target)

    measurement = perf.measure(task, target=target, workload=bench_config.workload, budget=budget.measure)
    if not measurement.done:
        return blocked("bench measurement failed", evidence=measurement)

    evidence = record_evidence(task, run=measurement.run, comparison=measurement.metrics, environment=budget.env)
    risks = [
        "benchmarks are not correctness proof",
        "uncontrolled variables can dominate",
        "ChunkedEP DeepEP claims require Slurm GPU evidence from the public bench",
    ]
    return done(bench=measurement.metrics, evidence=evidence, risks=risks)
```
