"""Pinned official forward execution with capture-only instrumentation.

This module never substitutes MLite arithmetic for official model methods.
The implemented execution profile is single-rank published forward only.
"""

import ast
import copy
import dataclasses
import hashlib
import itertools
import json
import os
import sys
import types
from pathlib import Path

import torch
from config_mapping import MAPPING, WAIVERS, map_release_config
from fixtures import (
    REFERENCE_SHA256,
    IdentityTokenizer,
    ownership_rows,
    reduced_overrides,
    validate_reference,
)
from megatron.lite.model.deepseek_v41.lite.checkpoint_store import validate_execution
from safetensors.torch import load_file

_MODULE_IDS = itertools.count()


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(item) for item in value)
    return value


def tensor_metadata(value):
    """Serialize capture descriptors only after the producing forward finishes."""
    if isinstance(value, torch.Tensor):
        raw = value.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            device=str(value.device),
            byte_length=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    if isinstance(value, dict):
        return {key: tensor_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_metadata(item) for item in value]
    return value


def forward_all_tokens(model, *args, _record_head=None, **kwargs):
    captured = []
    handle = model.head.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        original = model(*args, **kwargs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError(f"expected one backbone head invocation, got {len(captured)}")
    if _record_head is not None:
        _record_head(captured[0])
    full = model.head(captured[0], full_logits=True)
    return original, full


def validate_sequences(sequences, chunks, vocab_size, max_seq_len):
    names = []
    for sequence in sequences:
        if set(sequence) != {"id", "tokens"} or not isinstance(sequence["id"], str):
            raise ValueError("invalid sequence fields")
        name, tokens = sequence["id"], sequence["tokens"]
        if (
            not tokens
            or len(tokens) > max_seq_len
            or any(
                type(token) is not int or not 0 <= token < vocab_size
                for token in tokens
            )
        ):
            raise ValueError("invalid token sequence")
        if name in names or name not in chunks:
            raise ValueError("duplicate sequence or missing schedule")
        names.append(name)
        position = 0
        for index, chunk in enumerate(chunks[name]):
            if (
                set(chunk) != {"start_pos", "length"}
                or type(chunk["length"]) is not int
                or type(chunk["start_pos"]) is not int
            ):
                raise ValueError("invalid chunk fields")
            if (
                chunk["start_pos"] != position
                or chunk["length"] < 1
                or (index > 0 and chunk["length"] != 1)
            ):
                raise ValueError(
                    "chunks must be contiguous prefill then single-token decode"
                )
            position += chunk["length"]
        if position != len(tokens):
            raise ValueError("incomplete token coverage")
    if not names or set(names) != set(chunks):
        raise ValueError("sequence/schedule coverage mismatch")


class Recorder:
    def __init__(self, sequence_id):
        self.sequence_id = sequence_id
        self.layer = None
        self.chunk = 0
        self.records = []
        self.start_pos = 0
        self.length = 0
        self.occurrences = {}

    def add(self, stage, value):
        key = (self.chunk, self.layer, stage)
        occurrence = self.occurrences.get(key, 0)
        self.occurrences[key] = occurrence + 1
        owner = None if self.layer is None else ownership_rows()[self.layer]
        self.records.append(
            dict(
                sequence_id=self.sequence_id,
                chunk_id=self.chunk,
                layer_id=self.layer,
                stage=stage,
                occurrence=occurrence,
                owner=owner,
                query_positions=list(
                    range(self.start_pos, self.start_pos + self.length)
                ),
                value=snapshot(value),
            )
        )

    def index(self, stage, values):
        self.add(
            stage,
            {
                key: values[key]
                for key in ("q", "index_k", "index_score", "weights", "compress_lens")
                if key in values
            },
        )


def _official_module(reference, recorder=None):
    """Verify that instrumentation adds only recorder calls to Indexer.forward."""
    path = Path(reference) / "model.py"
    tree = ast.parse(path.read_text())
    original = ast.dump(tree, include_attributes=False)
    additions = []
    if recorder is not None:
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Indexer"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        body = []
        for statement in method.body:
            if isinstance(statement, ast.Return):
                call = ast.parse(
                    '__capture_index__("index.scores.final", locals())'
                ).body[0]
                body.append(call)
                additions.append(ast.unparse(call))
            body.append(statement)
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "index_score"
                for target in statement.targets
            ):
                call = ast.parse(
                    '__capture_index__("index.scores.intermediate", locals())'
                ).body[0]
                body.append(call)
                additions.append(ast.unparse(call))
        method.body = body
        stripped = copy.deepcopy(tree)

        class Strip(ast.NodeTransformer):
            def visit_Expr(self, node):
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "__capture_index__"
                ):
                    return None
                return self.generic_visit(node)

        stripped = Strip().visit(stripped)
        assert (
            ast.dump(stripped, include_attributes=False) == original
        ), "arithmetic AST changed"
    name = f"_ds41_official_{next(_MODULE_IDS)}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__["__capture_index__"] = None if recorder is None else recorder.index
    sys.modules[name] = module
    sys.path.insert(0, str(Path(reference).resolve()))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module, additions


class CaptureSession:
    def __init__(self, module, model, recorder):
        self.handles = []
        self.restores = []
        self.recorder = recorder
        r = recorder
        for i, block in enumerate(model.layers):

            def before(layer, inputs, i=i):
                r.layer = i
                r.add("block.input", inputs[0])
                r.add("block.pre_mix", inputs[2])
                if i == 0:
                    r.add("hc_expanded", inputs[0])
                    r.add("image_merged", inputs[0][:, :, 0])

            self.handles.append(block.register_forward_pre_hook(before))

            def after(layer, inputs, out):
                r.add("block.output", out[0])
                r.add("block.next_mix", out[1])

            self.handles.append(block.register_forward_hook(after))

            def attention_input(layer, inputs, i=i):
                r.add("attn.input", inputs[0])
                if i == 20:
                    r.add("ced.x20", inputs[0])

            self.handles.append(block.attn.register_forward_pre_hook(attention_input))
            self.handles.append(
                block.attn.register_forward_hook(
                    lambda layer, inputs, out: r.add("attn.output", out)
                )
            )
            self.handles.append(
                block.ffn.register_forward_pre_hook(
                    lambda layer, inputs: r.add("ffn.input", inputs[0])
                )
            )
            self.handles.append(
                block.ffn.register_forward_hook(
                    lambda layer, inputs, out: r.add("ffn.output", out)
                )
            )
            self.handles.append(
                block.ffn.gate.register_forward_hook(
                    lambda layer, inputs, out: r.add("ffn.routing", out)
                )
            )
            if block.engram is not None:
                self.handles.append(
                    block.engram.register_forward_pre_hook(
                        lambda layer, inputs, i=i: setattr(r, "layer", i)
                    )
                )
                self.handles.append(
                    block.engram.register_forward_hook(
                        lambda layer, inputs, out: r.add("engram.output", out)
                    )
                )
            if block.attn.compressor is not None:
                self.handles.append(
                    block.attn.compressor.register_forward_hook(
                        lambda layer, inputs, out: r.add(
                            "compressor.latent_pre_rope", out
                        )
                    )
                )
            if block.attn.indexer is not None and block.attn.indexer.owns_k:
                self.handles.append(
                    block.attn.indexer.k_norm.register_forward_hook(
                        lambda layer, inputs, out: r.add("index.k_pre_rope", out)
                    )
                )
        self.handles.append(
            model.embed.register_forward_hook(
                lambda layer, inputs, out: r.add("embedding", out)
            )
        )
        self.handles.append(
            model.engram_hash.register_forward_hook(
                lambda layer, inputs, out: r.add("engram.hash", out)
            )
        )
        for method, stage in (
            ("_window_kv", "swa.kv"),
            ("_compress_kv", "main.kv_published"),
        ):
            original = getattr(module.Attention, method)

            def wrapped(layer, *args, _original=original, _stage=stage, **kwargs):
                output = _original(layer, *args, **kwargs)
                r.add(_stage, output)
                if _stage == "main.kv_published":
                    r.add("topk", output[1])
                    r.add("candidates", module.shared_attn.candidates)
                    key = module.shared_attn.index_k
                    count = (args[2] + args[0].shape[1]) // layer.compress_ratio
                    r.add(
                        "index.k_published",
                        None if key is None else key[: args[0].shape[0], :count],
                    )
                    r.add(
                        "main.storage_identity", output[0].untyped_storage().data_ptr()
                    )
                return output

            self.restores.append((module.Attention, method, original))
            setattr(module.Attention, method, wrapped)
        original = module.sparse_attn

        def sparse(*args, **kwargs):
            r.add("sparse.args", args)
            output = original(*args, **kwargs)
            r.add("sparse.output", output)
            return output

        self.restores.append((module, "sparse_attn", original))
        module.sparse_attn = sparse

    def close(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, value in self.restores:
            setattr(owner, name, value)


@torch.inference_mode()
def _execute(
    reference,
    manifest,
    weights,
    sequence,
    chunks,
    device,
    seed,
    images,
    token_types,
    capture,
):
    recorder = Recorder(sequence["id"]) if capture else None
    module, additions = _official_module(reference, recorder)
    expected_args = map_release_config(
        manifest["original_config"], dataclasses.asdict(module.ModelArgs())
    )
    expected_args.update(reduced_overrides())
    expected_args["dspark_block_size"] = 0
    if expected_args != manifest["effective_model_args"]:
        raise ValueError("effective ModelArgs differ from explicit release mapping")
    torch.manual_seed(seed)
    module.shared_attn = module.SharedAttentionRuntime()
    with torch.device(device):
        model = module.Transformer(
            module.ModelArgs(**manifest["effective_model_args"]),
            tokenizer=IdentityTokenizer(),
        )
    params = dict(model.named_parameters())
    if set(params) != set(weights):
        raise ValueError("official binding key coverage mismatch")
    for name, target in params.items():
        source = weights[name]
        if target.shape != source.shape:
            raise ValueError(f"binding shape mismatch: {name}")
        if target.dtype == torch.float4_e2m1fn_x2:
            if source.dtype != torch.int8:
                raise ValueError(f"packed source must be I8: {name}")
            target.view(torch.uint8).copy_(source.view(torch.uint8))
        else:
            target.copy_(source)
        expected_bytes = (
            source.view(torch.uint8)
            if target.dtype == torch.float4_e2m1fn_x2
            else source.to(target.dtype).view(torch.uint8)
        )
        if not torch.equal(target.detach().view(torch.uint8).cpu(), expected_bytes):
            raise ValueError(f"bound payload differs: {name}")
    expert_probes = {}
    probe = (
        torch.arange(
            manifest["effective_model_args"]["dim"], device=device, dtype=torch.float32
        )[None, :]
        / 256
    ).bfloat16()
    for layer_id, block in enumerate(model.layers):
        for expert_id, expert in enumerate(block.ffn.experts):
            expert_probes[f"layers.{layer_id}.ffn.experts.{expert_id}"] = expert(
                probe
            ).clone()
    session = CaptureSession(module, model, recorder) if capture else None
    outputs = []
    last_outputs = []
    try:
        for index, chunk in enumerate(chunks):
            if recorder:
                recorder.chunk = index
                recorder.layer = None
                recorder.start_pos = chunk["start_pos"]
                recorder.length = chunk["length"]
            start, end = chunk["start_pos"], chunk["start_pos"] + chunk["length"]
            ids = torch.tensor(
                [sequence["tokens"][start:end]], dtype=torch.int64, device=device
            )
            types_ = (
                None
                if token_types is None
                else token_types[start:end].to(device)[None, :]
            )
            original, full = forward_all_tokens(
                model,
                ids,
                start_pos=start,
                images=[images] if images and index == 0 else None,
                token_types=types_,
                _record_head=(
                    None
                    if recorder is None
                    else lambda value: recorder.add("head.input", value)
                ),
            )
            if recorder:
                recorder.add("head.all_logits", full)
            if not torch.isfinite(full).all():
                raise ValueError("nonfinite official logits")
            if not torch.equal(full[:, -1], original[1]):
                error = (full[:, -1] - original[1]).abs().max().item()
                raise ValueError(
                    f"full-head/final-slice tolerance requires qualification: max_abs_error={error}"
                )
            outputs.append(full[0].clone())
            last_outputs.append(original[1][0].clone())
    finally:
        if session:
            session.close()
    if recorder:
        for record in recorder.records:
            record["rank"] = 0
            record["tensor_metadata"] = tensor_metadata(record["value"])
    return dict(
        logits=torch.cat(outputs),
        last_logits=last_outputs,
        captures=[] if recorder is None else recorder.records,
        binding_keys=sorted(params),
        ast_capture_additions=additions,
        expert_probes=expert_probes,
    )


def run_forward(request):
    required = {
        "schema_version",
        "reference",
        "fixture",
        "config",
        "weights",
        "sequences",
        "chunks",
        "images",
        "token_types",
        "execution",
        "capture",
        "enable_dspark_execution",
    }
    if (
        set(request) != required
        or request["schema_version"] != "deepseek-v41-oracle-v1"
    ):
        raise ValueError("unknown or missing oracle request fields")
    validate_execution(enable_dspark_execution=request["enable_dspark_execution"])
    if set(request["reference"]) != {"directory", "sha256"} or set(
        request["fixture"]
    ) != {"directory", "manifest_sha256"}:
        raise ValueError("unknown reference/fixture fields")
    reference = Path(request["reference"]["directory"])
    validate_reference(reference)
    if request["reference"]["sha256"] != REFERENCE_SHA256:
        raise ValueError("reference manifest mismatch")
    fixture = Path(request["fixture"]["directory"])
    raw = (fixture / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != request["fixture"]["manifest_sha256"]:
        raise ValueError("fixture manifest digest mismatch")
    manifest = json.loads(raw)
    if (
        manifest["profile"] != "reduced-forward-v1"
        or manifest["overrides"] != reduced_overrides()
    ):
        raise ValueError("unsupported fixture profile or dimensions")
    if manifest["owners"] != ownership_rows():
        raise ValueError("fixture owner graph mismatch")
    if manifest["original_config"] != json.loads(
        (reference / "config.json").read_text()
    ):
        raise ValueError("release config was rewritten")
    if manifest["effective_model_args"]["dspark_block_size"] != 0:
        raise ValueError("oracle constructor must not allocate DSpark")
    for key, value in reduced_overrides().items():
        if manifest["effective_model_args"][key] != value:
            raise ValueError(f"effective reduced dimension mismatch: {key}")
    if request["config"] != {
        "original": manifest["original_config"],
        "overrides": manifest["effective_model_args"],
    }:
        raise ValueError("explicit config mapping mismatch")
    if set(manifest["file_digests"]) != {
        "release.safetensors",
        "converted.safetensors",
        "inputs.safetensors",
    }:
        raise ValueError("unknown fixture payload files")
    for name, expected in manifest["file_digests"].items():
        if hashlib.sha256((fixture / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"fixture payload digest mismatch: {name}")
    if request["weights"] != {
        "file": "converted.safetensors",
        "sha256": manifest["file_digests"]["converted.safetensors"],
        "keys": manifest["converted_keys"],
    }:
        raise ValueError("weight manifest mismatch")
    execution = request["execution"]
    if (
        set(execution) != {"device", "seed", "rank", "world_size", "profile"}
        or execution["rank"] != 0
        or execution["world_size"] != 1
        or execution["profile"] != "published-forward-only"
    ):
        raise ValueError(
            "only single-rank published-forward-only execution is supported"
        )
    if request["capture"] != {"stages": "all"}:
        raise ValueError("complete capture coverage is mandatory")
    if not os.environ.get("SLURM_JOB_ID") or not execution["device"].startswith("cuda"):
        raise RuntimeError("official quantized oracle requires CUDA through Slurm")
    args = manifest["effective_model_args"]
    validate_sequences(
        request["sequences"], request["chunks"], args["vocab_size"], args["max_seq_len"]
    )
    sequence_ids = {sequence["id"] for sequence in request["sequences"]}
    if (set(request["images"]) | set(request["token_types"])) - sequence_ids:
        raise ValueError("images/types reference an unknown sequence")
    for sequence in request["sequences"]:
        name = sequence["id"]
        types_ = request["token_types"].get(name)
        images = request["images"].get(name, [])
        if types_ is not None and (
            types_.dtype != torch.int64
            or types_.shape != (len(sequence["tokens"]),)
            or not ((types_ >= -1) & (types_ <= 3)).all()
        ):
            raise ValueError("invalid token types")
        if images and types_ is None:
            raise ValueError("images require token types")
        covered = set()
        for image in images:
            start, end = image.start, image.start + image.types.numel()
            if (
                start < 0
                or end > request["chunks"][name][0]["length"]
                or covered.intersection(range(start, end))
            ):
                raise ValueError(
                    "image spans must be nonoverlapping and within prefill"
                )
            if (
                image.n_vit_h % 3
                or image.n_vit_w % 3
                or image.patches.shape != (image.n_vit_h * image.n_vit_w, 3, 14, 14)
            ):
                raise ValueError("invalid operator patch grid")
            if (image.types == 1).sum().item() != image.n_vit_h * image.n_vit_w // 9:
                raise ValueError("IMAGE slots do not match aligner rows")
            if not torch.equal(types_[start:end].cpu(), image.types.cpu()) or any(
                token != 255 for token in sequence["tokens"][start:end]
            ):
                raise ValueError("image span/types/token mismatch")
            covered.update(range(start, end))
        if types_ is not None and covered != {
            i for i, kind in enumerate(types_.tolist()) if kind >= 0
        }:
            raise ValueError("unbound image token types")
    weights = load_file(fixture / "converted.safetensors", device="cpu")
    results = []
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for sequence in request["sequences"]:
            name = sequence["id"]
            common = (
                reference,
                manifest,
                weights,
                sequence,
                request["chunks"][name],
                execution["device"],
                execution["seed"],
                request["images"].get(name),
                request["token_types"].get(name),
            )
            # The published launcher sets the default CUDA device for factories
            # in forward (notably cached SWA indices), as well as construction.
            # Scope it so the caller's default device is restored on failure.
            with torch.device(execution["device"]):
                baseline = _execute(*common, capture=False)
                instrumented = _execute(*common, capture=True)
            if not torch.equal(baseline["logits"], instrumented["logits"]):
                raise ValueError(
                    "instrumentation changed official outputs; tolerance qualification required"
                )
            for expert, value in baseline["expert_probes"].items():
                if not torch.equal(value, instrumented["expert_probes"][expert]):
                    raise ValueError(f"targeted expert execution changed: {expert}")
            instrumented["sequence_id"] = name
            results.append(instrumented)
    finally:
        torch.set_default_dtype(old_dtype)
    result_manifest = {
        **manifest,
        "execution": {
            **execution,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "input_sequences": request["sequences"],
        "chunks": request["chunks"],
        "config_leaf_mapping": MAPPING,
        "config_scope_waivers": WAIVERS,
    }
    return dict(
        manifest=result_manifest,
        sequences=results,
        captures=[capture for sequence in results for capture in sequence["captures"]],
        load_coverage={
            sequence["sequence_id"]: dict(
                keys=sequence["binding_keys"], bound_once=True, byte_comparison=True
            )
            for sequence in results
        },
        comparison={"baseline_instrumented_exact": True},
    )
