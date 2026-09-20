# Primitive Validate Skill

Validate one primitive before using it in a model.

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.validate", kind="state_machine", purpose="validate primitive correctness and precision",
    imports=["basic.constitution"], calls=["basic.construct_proxy_task", "basic.align_precision"],
    inputs=["task", "primitive", "implementation", "budget"],
    outputs=["validation", "evidence", "risks"], exits=["done", "blocked"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def validate(task, primitive, implementation, budget):
    proxy = basic.construct_proxy_task(
        task, target=primitive, reference=implementation.reference, variables=implementation.variables, budget=budget.proxy
    )
    if not proxy.done:
        return blocked("primitive proxy task not constructed", evidence=proxy)

    precision = basic.align_precision(task, target=primitive, variables=implementation.variables, budget=budget.precision)
    if not precision.done:
        return blocked("primitive precision not validated", evidence=precision)

    required = (
        "static_contract",
        "single_gpu_or_node_proxy",
        "controlled_variable_precision",
        "composition_with_adjacent_primitives",
        "usage_example_runs",
    )
    checks = getattr(implementation, "checks", {})
    if any(name not in checks for name in required):
        return blocked("missing executable validation checks", evidence=[name for name in required if name not in checks])
    validation = []
    for name in required:
        result = checks[name]()
        validation.append((name, result))
        if not result.done:
            return blocked("primitive check failed", evidence=validation)
    return done(validation=validation, evidence=[proxy, precision, validation], risks=precision.next)
```
