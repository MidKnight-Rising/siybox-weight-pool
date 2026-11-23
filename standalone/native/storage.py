"""Stream one trained native layer at a time through its original forward method."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_store import FaceStore, FlatStore, digest, save_face


def pack(checkpoint, root, layout):
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(root)
    trained = torch.load(checkpoint, map_location="cpu", weights_only=True)
    entries = {}
    with tempfile.TemporaryDirectory(dir=root.parent) as temp:
        stage = Path(temp) / "store"
        stage.mkdir()
        for i, (name, tensor) in enumerate(sorted(trained["state"].items())):
            if tensor.dtype != torch.float32:
                raise ValueError(f"Expected FP32: {name}")
            words = tensor.contiguous().numpy().view("<u4")
            stem = f"tensor-{i:03d}"
            entry = {"shape": list(words.shape), "file": stem, "sha256": digest(words)}
            if layout == "face":
                entry["full"] = save_face(stage, stem, words)
            else:
                np.save(stage / (stem + ".npy"), words)
            entries[name] = entry
        doc = {"format": "siybox-scalar-face-v1" if layout == "face" else "flat-fp32-v1",
               "arm": trained["arm"], "stream_sha256": trained["stream_sha256"],
               "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "tensors": entries}
        (stage / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n")
        reader = (FaceStore if layout == "face" else FlatStore)(stage, "cpu")
        for name, tensor in trained["state"].items():
            if not torch.equal(tensor.view(torch.int32), reader.get(name).view(torch.int32)):
                raise ValueError(f"Bitwise weight verification failed: {name}")
        del reader
        stage.rename(root)


def bind(model, root, layout):
    store = (FaceStore if layout == "face" else FlatStore)(root)
    if set(model.state_dict()) != set(store.entries):
        raise ValueError("Store does not match native model")
    for prefix, module in model.named_modules():
        path = prefix + "." if prefix else ""
        for local, _ in module.named_buffers(recurse=False):
            module._buffers[local] = store.get(path + local)
        local_names = tuple(module._parameters)
        if not local_names:
            continue
        names = {n: path + n for n in local_names if module._parameters[n] is not None}
        for n in names:
            module._parameters[n] = None

        def load(layer, inputs, mapping=names):
            for local, name in mapping.items():
                layer._parameters[local] = torch.nn.Parameter(store.get(name), requires_grad=False)

        def release(layer, inputs, output, mapping=names):
            for local in mapping:
                layer._parameters[local] = None

        module.register_forward_pre_hook(load)
        module.register_forward_hook(release, always_call=True)
    return store


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("layout", choices=("flat", "face"))
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--checkpoint", type=Path, required=True)
    args = cli.parse_args()
    pack(args.checkpoint, args.root, args.layout)
