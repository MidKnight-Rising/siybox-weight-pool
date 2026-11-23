"""Generate from the standalone scalar-face directory using greedy decoding."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from tokenizers import Tokenizer

from executor import Executor
from face_store import FaceStore


def generate(root, prompt, tokens, rows, direct):
    config = json.loads((root / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    input_ids = tokenizer.encode(prompt).ids
    if len(input_ids) + tokens > config["max_position_embeddings"]:
        raise ValueError("Prompt plus generation exceeds this checkpoint's context")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    executor = Executor(FaceStore(root), config, row_window=rows, direct=direct)
    ids = torch.tensor([input_ids], device="cuda")
    cache = None
    generated = []
    for _ in range(tokens):
        logits, cache = executor.forward(ids, cache)
        ids = logits[:, -1, :].argmax(-1, keepdim=True)
        token = ids.item()
        generated.append(token)
        del logits
        if token == config["eos_token_id"]:
            break
    return tokenizer.decode(input_ids + generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    if args.tokens < 1 or args.rows < 0:
        parser.error("tokens must be positive and rows nonnegative")
    print(generate(args.root, args.prompt, args.tokens, args.rows, args.direct))
