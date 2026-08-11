# uv WIP environment

This launcher layers a container-managed dependency site over live VERL,
Megatron-Core, and Megatron Lite source trees. It deliberately does not install
any of those source trees into the dependency site.

Set the four roots inside the container, then pass the command to execute:

```bash
export UV_WIP_SITE=/path/to/uv/site
export VERL_ROOT=/path/to/verl
export MEGATRON_ROOT=/path/to/Megatron-LM
export MLITE_ROOT=/path/to/Megatron-LM
# Optional torch-free kernel overlay. Its CUTLASS Python package is expanded
# explicitly because .pth files are not processed from PYTHONPATH entries.
export UV_WIP_KERNEL_SITE=/path/to/kernel/site-packages

bash experimental/lite/examples/verl/env/uv-wip/env.sh \
  /usr/bin/python3 -c 'import torch, vllm; import verl_mlite.engine'
```

The launcher validates identifying package directories and the uv site's
`.uv-wip-ready` publication marker before changing the environment. It
prepends, in order, the uv site, VERL source, Megatron source, Megatron Lite
package, and the local VERL integration package. Existing
`PYTHONPATH` entries remain last. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are
forced to `1` because the container's MKL path is not safe with the inherited
thread defaults used by the cluster launcher. `CC` and `CXX` are fixed to the
container's system compilers and the CUDA/system binary directories are
prepended to `PATH`; this prevents an inherited conda compiler from breaking
Triton's runtime helper build against `/usr/bin/python3`.

When `UV_WIP_KERNEL_SITE` is set, the launcher validates and appends both its
`nvidia_cutlass_dsl/python_packages` directory and the site root. They stay
after the required uv/VERL/Megatron/MLite source prefix but before inherited
`PYTHONPATH` entries.

The uv site is expected to be built inside the target container with
`/usr/bin/python3`. It must not contain distributions named `torch`,
`torchvision`, `torchaudio`, `torchcodec`, `torch-c-dlpack-ext`, `functorch`,
`triton`, or `nvidia-*`; those would override the CUDA and PyTorch stack
supplied by the base image.

`bootstrap.sh` enforces that contract while resolving the dependencies of an
explicit prebuilt vLLM wheel. Both the target and wheel URL are caller-owned so
the recipe contains no cluster-specific path:

```bash
export UV_WIP_SITE=/path/to/new/site
export VLLM_WHEEL_URL=https://wheels.vllm.ai/<commit>/<wheel>.whl
export UV_BIN=/path/to/uv
bash experimental/lite/examples/verl/env/uv-wip/bootstrap.sh
```

The target must not exist. The script first performs a dry resolution and
rejects a plan containing a container-owned package. It then installs into a
temporary sibling directory, audits distribution metadata, and rechecks the
native torch version. Only a successful audit writes `.uv-wip-ready` and
atomically renames that directory to the requested target; failures remove the
staging directory and leave no runnable target. `source-requirements.txt`
supplies VERL's runtime dependencies without installing VERL, Megatron-Core,
or Megatron Lite themselves.

The WIP source requirements intentionally use NumPy 2. vLLM main requires
`opencv-python-headless>=4.13`, which in turn requires NumPy 2, while VERL's
current package metadata still says `numpy<2`. Installing the online VERL tree
as a distribution would therefore be unsatisfiable; source import validation is
the compatibility gate for this environment.

The bootstrap also exposes vLLM's bundled `triton_kernels` directory under its
historical top-level import name. Main imports that name while loading `LLM`,
before VERL's process-local compatibility hook can create the same alias.
