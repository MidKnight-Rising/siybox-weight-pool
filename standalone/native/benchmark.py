"""Train and measure one native LM per process on the pinned Tiny Shakespeare text."""
import argparse
import collections
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch

CONTEXT = 128
WIDTH = 120
DEPTH = 3
BATCH = 8
SEED = 1337
STEPS = 400
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
MIB = 1024**2
SOURCE_HASHES = {
    "bench_rules.py": "247964f3e328944a36460ad7326555312f348337b69639127aafa79018255f64",
    "bench_agreement_vs_attention.py": "baf2876aac00da157f5ea31bd364c68c70bb1963a96e37b2163525eed22a662e",
    "agreement_attention.py": "5500d7f9c0175d229d0c77c11adcc9f8e436079f2fda76996f6ee05da00f1919",
    "cube_rules.py": "d3d20bcb6c91a1444b3a45086ba5ab27be61e74051c3a4e4b14eb605528822f5",
    "word_agreement_bench.py": "24a988009f4708ae4be48def46bcd1bd8c928a6413c7c0e484fe2ee453bbc196"
}


def corpus(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        raise ValueError("Expected the pinned Tiny Shakespeare corpus")
    words = re.findall(r"[a-z']+|[.,!?;:\n]", raw.decode().lower())
    cut = int(len(words) * 0.9)
    vocabulary = {"<unk>": 0}
    for word, _ in collections.Counter(words[:cut]).most_common(7999):
        vocabulary[word] = len(vocabulary)
    ids = torch.tensor([vocabulary.get(word, 0) for word in words])
    return ids[:cut], ids[cut:], len(vocabulary)


def configure():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False


def make_model(arm, vocab, source):
    # Source files remain in the original Siybox repository, along with provenance.
    for name, digest in SOURCE_HASHES.items():
        body = (source / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(body).hexdigest() != digest:
            raise ValueError(f"Unexpected original model source: {name}")
    sys.path.insert(0, str(source.resolve()))
    from bench_rules import LM, RuleMix
    from bench_agreement_vs_attention import LearnedAttn

    if arm == "agreement":
        def mixer():
            return RuleMix(WIDTH, CONTEXT, ["both", "neither", "near"])
    else:
        def mixer():
            return LearnedAttn(WIDTH, CONTEXT)
    return LM(mixer, vocab, CONTEXT, C=WIDTH, depth=DEPTH)


def train_model(arm, path, source, checkpoint):
    train, _, vocab = corpus(path)
    configure()
    model = make_model(arm, vocab, source).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    indices = torch.Generator().manual_seed(SEED)
    stream = hashlib.sha256()
    offsets = torch.arange(CONTEXT + 1)
    for step in range(STEPS):
        starts = torch.randint(0, len(train) - CONTEXT, (BATCH,), generator=indices)
        stream.update(starts.numpy().tobytes())
        data = train[starts[:, None] + offsets].cuda()
        for group in optimizer.param_groups:
            group["lr"] = 0.001 * min(1, (step + 1) / 20)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(data[:, :-1], data[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    cpu_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"arm": arm, "state": cpu_state, "stream_sha256": stream.hexdigest()}, checkpoint)


def share_fixed_buffers(model):
    """Transfer each distinct immutable inference buffer once across all blocks."""
    copies = {}
    for module in model.modules():
        for name, value in module.named_buffers(recurse=False):
            key = (value.dtype, value.shape, value.detach().contiguous().numpy().tobytes())
            if key not in copies:
                copies[key] = value.cuda()
            module._buffers[name] = copies[key]
    return len(copies)


def measure(arm, path, source, checkpoint, out, storage, root):
    train, val, vocab = corpus(path)
    configure()
    before_model = torch.cuda.memory_allocated()
    model = make_model(arm, vocab, source)
    count = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    distinct_buffers = None
    if storage in ("resident", "resident-shared"):
        trained = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if trained["arm"] != arm:
            raise ValueError("Checkpoint arm does not match")
        model.load_state_dict(trained["state"], strict=True)
        source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        stream_hash = trained["stream_sha256"]
        if storage == "resident-shared":
            distinct_buffers = share_fixed_buffers(model)
        model.cuda()
    else:
        from storage import bind
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["arm"] != arm:
            raise ValueError("Store arm does not match")
        source_hash = manifest["source_sha256"]
        stream_hash = manifest["stream_sha256"]
        bind(model, root, storage)
    model.eval()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    offsets = torch.arange(CONTEXT + 1)
    # Score fixed disjoint 128-target windows at their natural positions.
    total_loss = hits = 0
    logit_hash = hashlib.sha256()
    starts = torch.arange(0, len(val) - CONTEXT, CONTEXT)
    with torch.inference_mode():
        for chunk in starts.split(BATCH):
            block = val[chunk[:, None] + offsets].cuda()
            logits, loss = model(block[:, :-1], block[:, 1:])
            total_loss += loss.item() * len(chunk) * CONTEXT
            hits += (logits.argmax(-1) == block[:, 1:]).sum().item()
            logit_hash.update(memoryview(logits.detach().contiguous().cpu().numpy()).cast("B"))
        del block, logits, loss
        # Batch-one greedy generation, beginning with the same 128-token window.
        # Both original mixers recompute the context at every token.
        def generate(count):
            seq = val[:CONTEXT].unsqueeze(0).cuda()
            for _ in range(count):
                logits, _ = model(seq)
                next_id = logits[:, -1].argmax(-1, keepdim=True)
                seq = torch.cat((seq[:, 1:], next_id), dim=-1)
                del logits
            return seq
        generate(8)
        torch.cuda.synchronize()
        after_warmup = torch.cuda.memory_allocated()
        rates, peaks, reserved_peaks = [], [], []
        for _ in range(3):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            generate(64)
            torch.cuda.synchronize()
            rates.append(64 / (time.perf_counter() - started))
            peaks.append(torch.cuda.max_memory_allocated() / MIB)
            reserved_peaks.append(torch.cuda.max_memory_reserved() / MIB)
    scored = len(starts) * CONTEXT
    report = dict(arm=arm, storage=storage, distinct_shared_buffers=distinct_buffers,
                  architecture="native_three_rule_agreement" if arm == "agreement" else "single_head_eager_causal_attention",
                  model_parameters=count, width=WIDTH, depth=DEPTH, context=CONTEXT,
                  corpus_sha256=CORPUS_SHA256, train_tokens=len(train), validation_tokens=len(val), scored_tokens=scored, vocabulary=vocab,
                  train_steps=STEPS, train_batch=BATCH, training_targets=STEPS*BATCH*CONTEXT, seed=SEED,
                  training_start_offsets_sha256=stream_hash, checkpoint_sha256=source_hash, validation_logits_sha256=logit_hash.hexdigest(),
                  validation_logit_count=scored*vocab, validation_loss_nats=total_loss/scored,
                  validation_correct=hits, validation_accuracy=hits/scored, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  fp32=True, tf32=False, training_and_inference_in_separate_processes=True, before_model_allocated_mib=before_model/MIB,
                  parameter_bytes=parameter_bytes, buffer_bytes=buffer_bytes, live_model_allocated_mib=resident/MIB,
                  warm_idle_allocated_mib=after_warmup/MIB, decode_peak_allocated_mib=peaks,
                  decode_peak_reserved_mib=reserved_peaks, decode_tokens_per_second=rates,
                  median_decode_tokens_per_second=statistics.median(rates), context_recomputed=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "measure"))
    parser.add_argument("arm", choices=("agreement","attention"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--storage", choices=("resident", "resident-shared", "face", "flat"), default="resident")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.operation == "train" or args.storage in ("resident", "resident-shared"):
        if args.checkpoint is None:
            parser.error("supply --checkpoint for training or resident inference")
    elif not args.root:
        parser.error("supply --root for pooled inference")
    if args.operation == "train":
        train_model(args.arm, args.corpus, args.source, args.checkpoint)
    elif args.out is None:
        parser.error("measuring requires --out")
    else:
        measure(args.arm, args.corpus, args.source, args.checkpoint, args.out, args.storage, args.root)
