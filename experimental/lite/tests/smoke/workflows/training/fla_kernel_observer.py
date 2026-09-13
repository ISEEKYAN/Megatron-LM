"""Observe compiled FLA launches in Slurm probes; optional real arithmetic fault."""

import triton
from fla.modules.l2norm import l2norm_fwd_kernel
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from triton.compiler import ASTSource


class FLAKernelObserver:
    def __init__(self, mutate=False):
        self.mutate = mutate
        self.records = []

    def __enter__(self):
        self.original = CachingAutotuner.run
        observer = self

        def run(kernel, *args, **kwargs):
            name = kernel.fn.__name__
            if name not in ("l2norm_fwd_kernel", "l2norm_bwd_kernel"):
                return observer.original(kernel, *args, **kwargs)
            meta = kernel.triton_meta
            if observer.mutate and name == "l2norm_fwd_kernel":
                constants = dict(meta["constants"], BT=32)
                binary = triton.compile(
                    ASTSource(
                        l2norm_fwd_kernel.fn,
                        meta["signature"],
                        constexprs=constants,
                        attrs=meta["configs"][0],
                    ),
                    options={"num_warps": 8, "num_stages": 3, "enable_fp_fusion": True},
                )
                rows = int(args[4])
                result = binary[((rows + 31) // 32, 1, 1)](
                    *args[:5], constants["D"], constants["BD"], constants["NB"], 32
                )
            else:
                result = observer.original(kernel, *args, **kwargs)
                assert len(kernel.launchers) == 1, "FLA_ACTUAL_LAUNCHER_REQUIRED"
                winner = kernel.launchers[0].config
                matches = [r for r in kernel.compile_results if r.config == winner]
                assert len(matches) == 1, "FLA_ACTUAL_COMPILED_KERNEL_REQUIRED"
                binary = matches[0].kernel
            # Read executed binary options/constexprs, not policy or desired config.
            bt = binary.src.constants[(binary.src.fn.arg_names.index("BT"),)]
            config = dict(
                BT=bt,
                num_warps=binary.metadata.num_warps,
                num_stages=binary.metadata.num_stages,
            )
            names = (
                ("x", "y", "rstd")
                if name == "l2norm_fwd_kernel"
                else ("y", "rstd", "dy", "dx")
            )
            observer.records.append(
                dict(
                    name=name,
                    shape=list(args[0].shape),
                    dtype=str(args[0].dtype),
                    config=config,
                    kernel_hash=binary.hash,
                    tensors={n: t.detach().cpu().clone() for n, t in zip(names, args)},
                )
            )
            return result

        CachingAutotuner.run = run
        return self

    def __exit__(self, *exc):
        CachingAutotuner.run = self.original
