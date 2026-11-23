"""A two-fold, scalar-valued Siybox face store for FP32 checkpoints.

Each consecutive tile in a tensor is four 4x4x4 cubes. For each (row,column)
the outer face lists the occupied cubes in order. Each cube's cell lists
occupied inner layers and the FP32 words themselves. On disk the leaf order is
[tile,row,column,outer,inner]. A 16-bit occupancy mask per cell implies both
levels' counts and one-based tags. Entirely full tensors omit these masks.
An empty entry denotes FP32 +0. All other bit patterns, including -0, survive.

The flat comparator uses one ordinary, original-order NumPy array per tensor.
The stores never open a source checkpoint in inference.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch


SIDE = 4
TILE = SIDE ** 4
WORDS = np.dtype("<u4")
CHECKPOINT_SHA = "feb1cc5a5140f3ec7c357ecb9b13f54b17575d76c792459f563ce061a1527c6a"


def digest(words):
    return hashlib.sha256(memoryview(np.ascontiguousarray(words)).cast("B")).hexdigest()


def face_order(words):
    words = words.reshape(-1)
    pad = -len(words) % TILE
    if pad:
        words = np.pad(words, (0, pad))
    return words.reshape(-1, SIDE, SIDE, SIDE, SIDE).transpose(0, 3, 4, 1, 2).reshape(-1, TILE)


def tensor_order(words):
    return words.reshape(-1, SIDE, SIDE, SIDE, SIDE).transpose(0, 3, 4, 1, 2).reshape(-1)


def save_face(root, stem, original):
    cells = face_order(original)
    occupied = cells != 0
    full = bool(occupied.all())
    values = cells.ravel() if full else cells[occupied]
    np.save(root / (stem + ".npy"), values)
    if not full:
        mask = np.packbits(occupied, axis=1, bitorder="little")
        # Each offset points into this tensor's face values, never into an external file.
        offsets = np.empty(len(cells) + 1, dtype="<u8")
        offsets[0] = 0
        np.cumsum(occupied.sum(axis=1), out=offsets[1:])
        np.save(root / (stem + ".mask.npy"), mask)
        np.save(root / (stem + ".offsets.npy"), offsets)
    return full


class _Store:
    def __init__(self, root, device, layout):
        self.root = Path(root)
        self.device = torch.device(device)
        doc = json.loads((self.root / "manifest.json").read_text())
        if doc["format"] != layout:
            raise ValueError(f"Expected {layout} at {root}")
        self.entries = doc["tensors"]
        self.arrays = {}
        self.reads = 0
        self.bytes = 0  # Logical FP32 bytes returned to the executor.
        self.encoded_bytes_read = 0  # Logical bytes addressed in mapped payloads, excluding OS I/O.
        for name, meta in self.entries.items():
            stem = meta["file"]
            self.arrays[name] = tuple(
                np.load(self.root / (stem + suffix), mmap_mode="c", allow_pickle=False)
                for suffix in self._suffixes(meta)
            )

    def get(self, name):
        return self.get_rows(name, 0, self.entries[name]["shape"][0])

    def get_rows(self, name, start, stop):
        shape = self.entries[name]["shape"]
        if not 0 <= start <= stop <= shape[0]:
            raise IndexError(f"{name}: invalid first-axis slice {start}:{stop}")
        row_size = math.prod(shape[1:])
        words = self._range(name, start * row_size, stop * row_size)
        self.reads += 1
        self.bytes += words.nbytes
        array = words.view("<f4").reshape(stop - start, *shape[1:])
        return torch.from_numpy(array).to(self.device)

    def _suffixes(self, meta):
        return (".npy",)


class FlatStore(_Store):
    def __init__(self, root, device="cuda"):
        super().__init__(root, device, "flat-fp32-v1")

    def _range(self, name, begin, end):
        words = self.arrays[name][0].reshape(-1)[begin:end]
        self.encoded_bytes_read += words.nbytes
        return words


class FaceStore(_Store):
    def __init__(self, root, device="cuda"):
        super().__init__(root, device, "siybox-scalar-face-v1")

    def _suffixes(self, meta):
        return (".npy",) if meta["full"] else (".npy", ".mask.npy", ".offsets.npy")

    def get_face_tiles(self, name, first, limit):
        """Transfer complete physical face tiles for a kernel operating in face order.

        Dense tiles need only leaf values. Sparse tiles also carry little-endian
        occupancy bits and relative starting indices into the transferred leaves.
        """
        elements = math.prod(self.entries[name]["shape"])
        if not 0 <= first <= limit <= (elements + TILE - 1) // TILE:
            raise IndexError(f"{name}: invalid tile window {first}:{limit}")
        arrays = self.arrays[name]
        self.reads += 1
        self.bytes += max(0, min(elements, limit * TILE) - first * TILE) * WORDS.itemsize
        masks = offsets = None
        if len(arrays) == 1:
            words = arrays[0][first * TILE:limit * TILE]
        else:
            values, all_masks, all_offsets = arrays
            lo, hi = int(all_offsets[first]), int(all_offsets[limit])
            words = values[lo:hi]
            mask = all_masks[first:limit]
            prefix = all_offsets[first:limit + 1]
            relative = (prefix - lo).astype(np.int64)
            self.encoded_bytes_read += mask.nbytes + prefix.nbytes
            masks = torch.from_numpy(mask).to(self.device)
            offsets = torch.from_numpy(relative).to(self.device)
        self.encoded_bytes_read += words.nbytes
        leaves = torch.from_numpy(words.view("<f4")).to(self.device)
        return leaves, masks, offsets

    def _range(self, name, begin, end):
        if begin == end:
            return np.empty(0, dtype=WORDS)
        first = begin // TILE
        limit = (end + TILE - 1) // TILE
        arrays = self.arrays[name]
        if len(arrays) == 1:
            cells = arrays[0][first*TILE:limit*TILE]
            self.encoded_bytes_read += cells.nbytes
        else:
            values, masks, offsets = arrays
            mask = masks[first:limit]
            leaves = values[int(offsets[first]):int(offsets[limit])]
            bits = np.unpackbits(mask, bitorder="little").view(bool)
            cells = np.zeros(len(bits), dtype=WORDS)
            cells[bits] = leaves
            self.encoded_bytes_read += mask.nbytes + leaves.nbytes + 2*offsets.itemsize
        start = begin % TILE
        return tensor_order(cells)[start:start + end-begin]

    def cell(self, name, tile, row, column):
        """Read one physical face cell as the canonical nested Siybox list."""
        if not (0 <= row < SIDE and 0 <= column < SIDE):
            raise IndexError((row, column))
        if not 0 <= tile < (math.prod(self.entries[name]["shape"]) + TILE - 1) // TILE:
            raise IndexError(tile)
        arrays = self.arrays[name]
        cell_size = SIDE * SIDE
        pos = (row * SIDE + column) * cell_size
        if len(arrays) == 1:
            start = tile*TILE + pos
            words = arrays[0][start:start+cell_size]
        else:
            values, masks, offsets = arrays
            bits = np.unpackbits(masks[tile], bitorder="little").view(bool)
            start = int(offsets[tile]) + int(bits[:pos].sum())
            mask = bits[pos:pos+cell_size]
            words = np.zeros(cell_size, dtype=WORDS)
            words[mask] = values[start:start+int(mask.sum())]
        entries = []
        for outer, sub in enumerate(words.reshape(SIDE, SIDE), 1):
            leaves = [(depth, int(word)) for depth, word in enumerate(sub, 1) if word]
            if leaves:
                entries.append((outer, [len(leaves), *leaves]))
        return [len(entries), *entries]


def pack(model, faces, flat):
    from safetensors import safe_open  # Conversion-only dependency.
    if faces.exists() or flat.exists() or faces.resolve() == flat.resolve():
        raise FileExistsError("Use two distinct, new output directories")
    source = model / "model.safetensors"
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if source_hash != CHECKPOINT_SHA:
        raise ValueError("This test requires the published TinyMoE FP32 checkpoint")
    ref_path = Path(__file__).resolve().parents[1] / "siybox.py"
    spec = importlib.util.spec_from_file_location("reference_siybox", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    face_entries, flat_entries = {}, {}
    faces.parent.mkdir(parents=True, exist_ok=True)
    flat.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=faces.parent) as face_tmp, tempfile.TemporaryDirectory(dir=flat.parent) as flat_tmp:
        # Separate child dirs allow atomic renames out of each temporary directory.
        froot, troot = Path(face_tmp)/"pack", Path(flat_tmp)/"pack"
        froot.mkdir()
        troot.mkdir()
        parameters = 0
        with safe_open(source, framework="numpy", device="cpu") as archive:
            for i, name in enumerate(sorted(archive.keys())):
                original = archive.get_tensor(name)
                if original.dtype != np.dtype("<f4"):
                    raise TypeError(name)
                words = original.view(WORDS)
                stem = f"tensor-{i:03d}"
                meta = {"shape": list(original.shape), "file": stem, "sha256": digest(words)}
                flat_entries[name] = meta
                np.save(troot / (stem + ".npy"), words)
                face_entries[name] = {**meta, "full": save_face(froot, stem, words)}
                parameters += original.size
        for root, layout, entries in [(froot,"siybox-scalar-face-v1",face_entries),(troot,"flat-fp32-v1",flat_entries)]:
            doc = {"format":layout,"source_sha256":source_hash,"tensors":entries}
            (root / "manifest.json").write_text(json.dumps(doc, indent=2)+"\n")
            for filename in ("config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","generation_config.json"):
                shutil.copyfile(model/filename,root/filename)

        face = FaceStore(froot,"cpu")
        conventional = FlatStore(troot,"cpu")
        cells_checked = 0
        with safe_open(source, framework="numpy", device="cpu") as archive:
            for name in sorted(archive.keys()):
                orig = archive.get_tensor(name).view(WORDS)
                for store in (face, conventional):
                    actual = store.get(name).numpy().view(WORDS)
                    if not np.array_equal(orig, actual):
                        raise AssertionError(name)
                    # Verify first/last rows and a nonzero offset against the source.
                    for row in {0, orig.shape[0] // 2, orig.shape[0]-1}:
                        if not np.array_equal(store.get_rows(name,row,row+1).numpy().view(WORDS), orig[row:row+1]):
                            raise AssertionError((name,row))
                tile_count = (orig.size + TILE - 1) // TILE
                for index in {0,tile_count-1}:
                    words = orig.ravel()[index*TILE:(index+1)*TILE]
                    padded = np.pad(words, (0, TILE - len(words))).reshape(SIDE,SIDE,SIDE,SIDE)
                    expected = ref.encode_faces([ref.encode(cube.tolist()) for cube in padded])
                    for row in range(SIDE):
                        for column in range(SIDE):
                            if face.cell(name,index,row,column) != expected[row][column]:
                                raise AssertionError((name,index,row,column))
                            cells_checked += 1

        receipt = {"parameters": parameters,"source_sha256":source_hash,"tensors_verified_each_layout":len(face_entries),
                   "nested_cells_compared_to_siybox_reference":cells_checked,
                   "full_tensors":sum(e["full"] for e in face_entries.values()),
                   "face_weight_files_bytes":sum(p.stat().st_size for p in froot.glob("*.npy")),
                   "flat_weight_files_bytes":sum(p.stat().st_size for p in troot.glob("*.npy")),
                   "face_manifest_bytes":(froot/"manifest.json").stat().st_size,
                   "flat_manifest_bytes":(troot/"manifest.json").stat().st_size}
        for root in (froot,troot):
            (root/"pack-result.json").write_text(json.dumps(receipt,indent=2)+"\n")
        froot.rename(faces)
        troot.rename(flat)
        print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("operation",choices=["pack"])
    cli.add_argument("--model",required=True,type=Path)
    cli.add_argument("--faces",required=True,type=Path)
    cli.add_argument("--flat",required=True,type=Path)
    args = cli.parse_args()
    pack(args.model,args.faces,args.flat)
