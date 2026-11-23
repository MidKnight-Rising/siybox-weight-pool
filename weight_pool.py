"""Run one original Mixtral checkpoint through a Siybox layer-list weight pool."""
import argparse
import hashlib
import importlib.util
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.activations import ACT2FN


MODEL = "FlameF0X/TinyMoE-100m-2x8-retrained"
REVISION = "0c1d097c964b949bc0092fa8e99c88c325421167"
CHECKPOINT_SHA = "feb1cc5a5140f3ec7c357ecb9b13f54b17575d76c792459f563ce061a1527c6a"
EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)")
PROJECTIONS = ("gate_up_proj", "down_proj")
MODEL_SETTINGS = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
MIB = 2**20
PROMPTS = (
    "The capital of France is",
    "In Python, a function that reverses a string can be written as",
    "The small robot stumbled and then",
    "At night, the astronomer pointed her telescope",
    "Ingredients for a simple vegetable soup:",
    "To prove that the square root of two is irrational, suppose",
)


def write_json(path, contents):
    path.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_siybox(path):
    spec = importlib.util.spec_from_file_location("siybox_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_original(path):
    return AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager", experts_implementation="eager",
    ).eval()


def tensor_bytes(tensor):
    return memoryview(tensor.detach().contiguous().numpy()).cast("B")


def check_checkpoint_conversion(source, state):
    """Prove the pool covers all source weights, including fused expert inputs."""
    with safe_open(source, framework="pt", device="cpu") as original:
        keys = set(original.keys())
        expected = {}
        for name, tensor in state.items():
            match = EXPERT.fullmatch(name)
            if not match:
                expected[name.replace(".mlp.gate.", ".block_sparse_moe.gate.")] = tensor
                continue
            layer, projection = match.groups()
            for expert, block in enumerate(tensor):
                prefix = f"model.layers.{layer}.block_sparse_moe.experts.{expert}."
                if projection == "gate_up_proj":
                    expected[prefix + "w1.weight"], expected[prefix + "w3.weight"] = block.chunk(2)
                else:
                    expected[prefix + "w2.weight"] = block

        if keys != expected.keys():
            raise RuntimeError(f"Checkpoint key mismatch: {keys ^ expected.keys()}")
        for name, tensor in expected.items():
            if tensor_bytes(original.get_tensor(name)) != tensor_bytes(tensor):
                raise RuntimeError(f"Changed source bytes: {name}")
        return len(keys)


def pack(model_path, pool_path, siybox_path):
    reference = load_siybox(siybox_path)
    source = model_path / "model.safetensors"
    source_sha = sha256_file(source)
    if source_sha != CHECKPOINT_SHA:
        raise RuntimeError(f"Expected the pinned checkpoint, got {source_sha}")
    model = load_original(model_path)
    state = model.state_dict()
    source_keys = check_checkpoint_conversion(source, state)
    common = sorted(name for name in state if not EXPERT.fullmatch(name))
    side = math.ceil(math.sqrt(len(common)))
    common_grid = [[0] * side for _ in range(side)]
    layer_count = model.config.num_hidden_layers
    expert_count = model.config.num_local_experts
    layers = [[[[0, 0], [0, 0]] for _ in range(expert_count)] for _ in range(layer_count)]

    pool_path.mkdir(parents=True, exist_ok=True)
    settings = pool_path / "model-settings"
    settings.mkdir(exist_ok=True)
    for name in MODEL_SETTINGS:
        shutil.copyfile(model_path / name, settings / name)
    payload_name = pool_path / "weights.bin"
    with payload_name.open("wb") as payload:
        def append(tensor):
            if tensor.dtype != torch.float32:
                raise RuntimeError(f"Expected float32, got {tensor.dtype}")
            body = tensor_bytes(tensor)
            span = [payload.tell(), list(tensor.shape), len(body), hashlib.sha256(body).hexdigest()]
            payload.write(body)
            return span

        for index, name in enumerate(common):
            row, column = divmod(index, side)
            common_grid[row][column] = append(state[name])
        for layer_index, grids in enumerate(layers):
            for expert_index, grid in enumerate(grids):
                for projection_index, projection in enumerate(PROJECTIONS):
                    name = f"model.layers.{layer_index}.mlp.experts.{projection}"
                    grid[0][projection_index] = append(state[name][expert_index])

    manifest = {
        "format": 1, "source": MODEL, "revision": REVISION, "source_sha256": source_sha,
        "source_tensors_verified": source_keys, "siybox_sha256": sha256_file(siybox_path),
        "settings_sha256": {name: sha256_file(settings / name) for name in MODEL_SETTINGS},
        "layer_count": layer_count, "expert_count": expert_count,
        "parameters": sum(t.numel() for t in state.values()),
        "common_names": common,
        "common_face": reference.encode([common_grid]),
        "expert_face": reference.encode_faces([reference.encode(grids) for grids in layers]),
        "payload_bytes": payload_name.stat().st_size,
        "payload_sha256": sha256_file(payload_name),
    }
    write_json(pool_path / "faces.json", manifest)
    with FacePool(pool_path, mode="cube") as pool:
        for row in pool.manifest["common_face"]:
            for cell in row:
                if cell[0]:
                    pool.verify_span(cell[1][1])
        for layer in range(layer_count):
            for expert in range(expert_count):
                for projection in range(len(PROJECTIONS)):
                    pool.verify_span(pool.expert_span(layer, expert, projection))
    print(f"Packed and checked {manifest['parameters']:,} parameters across {source_keys} source tensors.")


def at_depth(cell, depth):
    """Direct ordinal lookup in this fully occupied layer list."""
    layer, value = cell[depth + 1]
    if layer != depth + 1:
        raise RuntimeError("Face entry does not match its expected one-based depth")
    return value


class FacePool:
    def __init__(self, root, mode):
        self.manifest = json.loads((root / "faces.json").read_text(encoding="utf-8"))
        if self.manifest["format"] != 1:
            raise RuntimeError("Unexpected face-pool format")
        self.handle = (root / "weights.bin").open("rb")
        self.payload = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_COPY)
        if len(self.payload) != self.manifest["payload_bytes"]:
            raise RuntimeError("Weight pool length differs from its manifest")
        self.mode = "cube"
        if mode == "flat":
            self.flat_experts = {
                (layer, expert, projection): self.expert_span(layer, expert, projection)
                for layer in range(self.manifest["layer_count"])
                for expert in range(self.manifest["expert_count"])
                for projection in range(len(PROJECTIONS))
            }
        self.mode = mode
        self.reads = 0
        self.bytes = 0
        self.decoded_experts = set()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.payload.close()
        self.handle.close()

    def common(self, device):
        face = self.manifest["common_face"]
        for index, name in enumerate(self.manifest["common_names"]):
            row, column = divmod(index, len(face))
            yield name, self.tensor(at_depth(face[row][column], 0), device)

    def expert_span(self, layer, expert, projection):
        if self.mode == "flat":
            return self.flat_experts[layer, expert, projection]
        layer_face_cell = at_depth(self.manifest["expert_face"][0][projection], layer)
        return at_depth(layer_face_cell, expert)

    def expert(self, layer, expert, device):
        self.decoded_experts.add((layer, expert))
        return tuple(self.tensor(self.expert_span(layer, expert, projection), device)
                     for projection in range(len(PROJECTIONS)))

    def tensor(self, span, device):
        start, shape, byte_count, _ = span
        self.reads += 1
        self.bytes += byte_count
        return torch.frombuffer(
            self.payload, dtype=torch.float32, count=byte_count // 4, offset=start,
        ).reshape(shape).to(device=device, copy=True)

    def verify_span(self, span):
        start, _, byte_count, digest = span
        result = hashlib.sha256(memoryview(self.payload)[start:start + byte_count]).hexdigest()
        if result != digest:
            raise RuntimeError(f"Invalid tensor bytes at {start}")


class PooledExperts(nn.Module):
    """Preserve Transformers' eager expert math; fetch a slab after native routing."""
    def __init__(self, config, pool, layer):
        super().__init__()
        self.pool = pool
        self.layer = layer
        self.experts = config.num_local_experts
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.experts).permute(2, 1, 0)
        selected = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        for expert in selected:
            gate_up, down = self.pool.expert(self.layer, expert, hidden_states.device)
            top_k_pos, token_idx = torch.where(mask[expert])
            gate, up = F.linear(hidden_states[token_idx], gate_up).chunk(2, dim=-1)
            contribution = F.linear(self.activation(gate) * up, down)
            contribution = contribution * top_k_weights[token_idx, top_k_pos, None]
            result.index_add_(0, token_idx, contribution.to(result.dtype))
            del gate_up, down
        return result


def load_pooled(model_path, pool):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    # Construct on CPU and remove the placeholder expert slabs before any CUDA move.
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="eager", experts_implementation="eager",
    ).eval()
    for index, layer in enumerate(model.model.layers):
        layer.mlp.experts = PooledExperts(config, pool, index)
    model.load_state_dict(dict(pool.common("cpu")), strict=True, assign=True)
    return model.to("cuda")


def gpu_reset():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def metrics():
    return {
        "allocated_mib": torch.cuda.memory_allocated() / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
    }


def inference(model, tokenizer, steps, pool):
    sequences = []
    scores = []
    prefill_time = decode_time = prefill_peak = decode_peak = 0
    prefill_experts = []
    decode_experts = []
    for text in PROMPTS:
        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        cache = None
        generated = []
        for step in range(steps):
            gpu_reset()
            reads_before = pool.reads
            started = time.perf_counter()
            result = model(ids, past_key_values=cache, use_cache=True, output_router_logits=False)
            next_logits = result.logits[:, -1, :].contiguous()
            ids = next_logits.argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / MIB
            experts = (pool.reads - reads_before) // len(PROJECTIONS)
            if step == 0:
                prefill_time += elapsed
                prefill_peak = max(prefill_peak, peak)
                prefill_experts.append(experts)
            else:
                decode_time += elapsed
                decode_peak = max(decode_peak, peak)
                decode_experts.append(experts)
            cache = result.past_key_values
            scores.append(next_logits.cpu())
            generated.append(ids.item())
            del result, next_logits
        sequences.append({"prompt": text, "tokens": generated,
                          "text": tokenizer.decode(generated, skip_special_tokens=True)})
        del cache, ids

    return sequences, torch.cat(scores), {
        "prefill_peak_allocated_mib": prefill_peak,
        "decode_peak_allocated_mib": decode_peak,
        "prefill_seconds": prefill_time, "decode_seconds": decode_time,
        "decode_tokens_per_second": len(PROMPTS) * (steps - 1) / decode_time,
        "prefill_expert_counts": prefill_experts if pool.reads else None,
        "decode_experts_min": min(decode_experts) if pool.reads else None,
        "decode_experts_max": max(decode_experts) if pool.reads else None,
    }


def run(model_path, pool_path, out, mode, steps):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with torch.inference_mode(), FacePool(pool_path, mode) as pool:
        source = model_path if mode == "resident" else pool_path / "model-settings"
        if mode != "resident":
            for name, digest in pool.manifest["settings_sha256"].items():
                if sha256_file(source / name) != digest:
                    raise RuntimeError(f"Changed packaged model settings: {name}")
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        gpu_reset()
        model = load_original(source).to("cuda") if mode == "resident" else load_pooled(source, pool)
        torch.cuda.synchronize()
        before = metrics()
        initial_bytes, initial_reads = pool.bytes, pool.reads
        sequences, scores, infer = inference(model, tokenizer, steps, pool)
        report = {
            "mode": mode, "gpu": torch.cuda.get_device_name(),
            "steps_per_prompt": steps, "parameters": pool.manifest["parameters"],
            "checkpoint_sha256": pool.manifest["source_sha256"],
            "pool_payload_sha256": pool.manifest["payload_sha256"],
            "original_checkpoint_used": mode == "resident",
            "model_resident": before,
            "initial_pool_bytes": initial_bytes, "initial_pool_reads": initial_reads,
            "inference_pool_bytes": pool.bytes - initial_bytes,
            "inference_pool_reads": pool.reads - initial_reads,
            "unique_layer_experts_used": len(pool.decoded_experts),
            "siybox_face_reads_inference": 2 * (pool.reads - initial_reads) if mode == "cube" else 0,
            "inference": infer, "sequences": sequences,
        }
        del model
    write_json(out / f"{mode}.json", report)
    torch.save(scores, out / f"{mode}-logits.pt")
    print(f"{mode}: resident {before['allocated_mib']:.1f} MiB, decode peak {infer['decode_peak_allocated_mib']:.1f} MiB, {infer['decode_tokens_per_second']:.1f} t/s")


def compare(out):
    reports = {mode: json.loads((out / f"{mode}.json").read_text(encoding="utf-8"))
               for mode in ("resident", "flat", "cube")}
    baseline = torch.load(out / "resident-logits.pt", weights_only=True)
    comparisons = {}
    for mode in ("flat", "cube"):
        logits = torch.load(out / f"{mode}-logits.pt", weights_only=True)
        comparisons[mode] = {
            "bit_equal_all_logits": torch.equal(baseline.view(torch.int32), logits.view(torch.int32)),
            "max_abs_logit_error": (baseline - logits).abs().max().item(),
            "all_generated_tokens_equal": reports["resident"]["sequences"] == reports[mode]["sequences"],
        }
    summary = {"model": MODEL, "revision": REVISION, "logit_rows": len(baseline),
               "logits_per_row": baseline.shape[-1], "comparisons": comparisons, "arms": reports}
    write_json(out / "RESULT.json", summary)
    print(json.dumps(comparisons, indent=2))
    if not all(item["bit_equal_all_logits"] and item["all_generated_tokens_equal"]
               for item in comparisons.values()):
        raise SystemExit("Output parity did not pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pack", "resident", "flat", "cube", "compare"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--siybox", type=Path)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.operation in ("pack", "resident") and args.model is None:
        parser.error(f"{args.operation} requires --model")
    if args.operation != "compare" and args.pool is None:
        parser.error(f"{args.operation} requires --pool")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.operation == "pack":
        if args.siybox is None:
            parser.error("pack requires --siybox")
        pack(args.model, args.pool, args.siybox)
    elif args.operation == "compare":
        compare(args.out)
    else:
        run(args.model, args.pool, args.out, args.operation, args.steps)


if __name__ == "__main__":
    main()
