"""One fresh process per arm. Compare with the original fully resident checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from tokenizers import Tokenizer


PROMPTS = (
    "The capital of France is",
    "In Python, a function that reverses a string can be written as",
    "The small robot stumbled and then",
    "At night, the astronomer pointed her telescope",
    "Ingredients for a simple vegetable soup:",
    "To prove that the square root of two is irrational, suppose",
)
ARMS = ("resident", "flat", "face")
MIB = 1024 ** 2


def label(arm, rows):
    return arm if not rows else f"{arm}-rows{rows}"


def load_runtime(args):
    if args.arm == "resident":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.root, dtype=torch.float32, local_files_only=True,
            attn_implementation="eager", experts_implementation="eager",
        ).eval().to("cuda")

        def forward(ids, cache):
            result = model(ids, past_key_values=cache, use_cache=True, output_router_logits=False)
            return result.logits, result.past_key_values

        return forward, None

    from executor import Executor
    from face_store import FaceStore, FlatStore
    store_class = FaceStore if args.arm == "face" else FlatStore
    store = store_class(args.root, device="cuda")
    config = json.loads((args.root / "config.json").read_text())
    model = Executor(store, config, row_window=args.rows, direct=args.direct)
    return model.forward, model


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    start_load = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(args.root / "tokenizer.json"))
    forward, executor = load_runtime(args)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start_load
    setup_peak = torch.cuda.max_memory_allocated() / MIB
    initial = torch.cuda.memory_allocated() / MIB
    scores, sequences = [], []
    stats = {p: {"peak_allocated_mib": 0, "peak_reserved_mib": 0, "seconds": 0, "steps": 0}
             for p in ["prefill", "decode"]}
    groups = set()
    decode_counts = set()
    for prompt in PROMPTS:
        input_tokens = tokenizer.encode(prompt).ids
        ids = torch.tensor([input_tokens], dtype=torch.long, device="cuda")
        cache = None
        output_tokens = []
        for step in range(args.steps):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            logits, cache = forward(ids, cache)
            next_logits = logits[:, -1, :].contiguous()
            ids = next_logits.argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            row = stats["decode" if step else "prefill"]
            row["seconds"] += elapsed
            row["steps"] += 1
            row["peak_allocated_mib"] = max(row["peak_allocated_mib"], torch.cuda.max_memory_allocated()/MIB)
            row["peak_reserved_mib"] = max(row["peak_reserved_mib"], torch.cuda.max_memory_reserved()/MIB)
            if executor is not None:
                routes = {(l, e) for l, experts in enumerate(executor.last_routes) for e in experts}
                groups.update(routes)
                if step:
                    decode_counts.add(len(routes))
            scores.append(next_logits.cpu())
            output_tokens.append(ids.item())
            del logits, next_logits
        sequences.append({"prompt": prompt, "input_tokens": input_tokens, "tokens": output_tokens})
        del cache, ids
    scores = torch.cat(scores).contiguous()
    imported = "transformers" in sys.modules
    if executor is not None and imported:
        raise RuntimeError("Standalone runtime unexpectedly imported Transformers")
    mode = label(args.arm, args.rows) + ("-direct" if args.direct else "")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(scores, args.out / f"{mode}-logits.pt")
    result = {
        "arm": mode, "gpu": torch.cuda.get_device_name(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "transformers_imported": imported, "load_seconds": load_seconds,
        "load_peak_allocated_mib": setup_peak, "initial_allocated_mib": initial,
        "inference": stats, "decode_tokens_per_second": stats["decode"]["steps"]/stats["decode"]["seconds"],
        "process_max_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "store_requests": executor.store.reads if executor else None,
        "direct_kernel_weight_reads": executor.direct.calls if executor and executor.direct else None,
        "fp32_bytes_requested": executor.store.bytes if executor else None,
        "encoded_bytes_addressed": executor.store.encoded_bytes_read if executor else None,
        "unique_layer_experts": len(groups) if executor else None,
        "decode_expert_counts": sorted(decode_counts) if executor else None,
        "logits_sha256": hashlib.sha256(scores.numpy().tobytes()).hexdigest(), "sequences": sequences,
    }
    (args.out / f"{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(mode, stats["decode"]["peak_allocated_mib"], result["decode_tokens_per_second"], flush=True)


def compare(args):
    baseline = torch.load(args.out / "resident-logits.pt", weights_only=True)
    records = {}
    for path in sorted(args.out.glob("*-logits.pt")):
        arm = path.name.removesuffix("-logits.pt")
        report = json.loads((args.out / f"{arm}.json").read_text())
        ref = json.loads((args.out / "resident.json").read_text())
        scores = torch.load(path, weights_only=True)
        if scores.shape != baseline.shape or scores.dtype != torch.float32 or baseline.dtype != torch.float32:
            raise ValueError(f"Incompatible logit arrays: {arm}")
        if [s["input_tokens"] for s in report["sequences"]] != [s["input_tokens"] for s in ref["sequences"]]:
            raise ValueError(f"Input tokens differ: {arm}")
        same_bits = baseline.view(torch.int32) == scores.view(torch.int32)
        report["comparison"] = {
            "bit_equal_logits": int(same_bits.sum()), "logits_compared": scores.numel(),
            "max_abs_logit_error": float((scores-baseline).abs().max()),
            "identical_generation": ref["sequences"] == report["sequences"],
        }
        records[arm] = report
    if len(records) < 2:
        raise ValueError("Need resident and at least one candidate")
    (args.out / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    for arm, report in records.items():
        print(arm, report["comparison"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=(*ARMS, "compare"))
    parser.add_argument("--root", type=Path, help="One checkpoint or one independent store")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=0, help="Embedding/head row window, 0 for full matrices")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--direct", action="store_true", help="Run weight arithmetic on encoded values in the GPU kernel")
    args = parser.parse_args()
    if args.arm != "compare" and (not args.root or args.steps < 2 or args.rows < 0):
        parser.error("supply --root, steps >= 2, rows >= 0")
    if args.arm == "resident" and (args.rows or args.direct):
        parser.error("row windows apply to streaming arms")
    compare(args) if args.arm == "compare" else run(args)
