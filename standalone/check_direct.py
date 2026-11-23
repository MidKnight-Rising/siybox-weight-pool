"""Perturb one scalar face leaf in a copy-on-write mapping and rerun a prompt."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from tokenizers import Tokenizer

from executor import Executor
from face_store import FaceStore, TILE


class DirectOnlyStore(FaceStore):
    def _range(self, name, begin, end):
        raise AssertionError(f"Direct execution tried reconstructing a tensor from {name}")


def check(root):
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((root / "config.json").read_text())
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    prompt = "A copper kettle sat on the stove"
    ids = torch.tensor([tok.encode(prompt).ids], device="cuda")
    store = DirectOnlyStore(root)
    model = Executor(store, cfg, row_window=2048, direct=True)

    def scores():
        return model.forward(ids)[0][0, -1].cpu()

    reference = scores()
    row = reference.argmax().item()
    name = "lm_head.weight"
    words, = store.arrays[name]  # This checkpoint's output head is fully occupied.
    if words.mode != "c":
        raise ValueError("This intervention requires a private copy-on-write mapping")
    width = store.entries[name]["shape"][1]
    tile, position = divmod(row * width, TILE)
    cell, depth = position % 16, position // 16
    offset = tile * TILE + cell * 16 + depth
    values = words.view("<f4")
    original = values[offset].copy()
    try:
        values[offset] = original + np.float32(1.0)
        perturbed = scores()
    finally:
        values[offset] = original
    restored = scores()
    changed = (reference.view(torch.int32) != perturbed.view(torch.int32)).nonzero().flatten().tolist()
    equal = torch.equal(reference.view(torch.int32), restored.view(torch.int32))
    if changed != [row] or not equal:
        raise AssertionError((row, changed, equal))
    return {"prompt": prompt, "tensor": name, "output_row": row, "weight_column": 0,
            "tile": tile, "face_cell": cell, "depth_index": depth, "changed_logit_indices": changed,
            "delta": (perturbed[row]-reference[row]).item(), "logits_checked": reference.numel(),
            "all_bits_restored": equal, "dense_reconstruction_blocked": True,
            "restored_logits_sha256": hashlib.sha256(restored.numpy().tobytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = json.dumps(check(args.root), indent=2) + "\n"
    if args.out:
        args.out.write_text(report)
    print(report)
