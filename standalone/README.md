# Standalone Siybox weight and inference experiment

**One complete open TinyMoE checkpoint ran directly from scalar-valued Siybox faces.** On an RTX 2070 SUPER, peak PyTorch-allocated GPU memory during decoding fell from **413.46 MiB** for the resident model to **36.23 MiB** for direct face arithmetic. All 144 generated tokens matched. A flat array using the same schedule and GPU kernel budget also reached **36.23 MiB**. The experiment demonstrates independent inference from face values and a large saving from streaming. It found no additional VRAM saving attributable to face ordering. [Raw measurements](results/comparison.json) · [Native Siybox language-model experiment](native/README.md)

| Same 99.8M FP32 checkpoint | Peak decode (MiB) | Decode tokens/s | Output versus resident |
|---|---:|---:|---|
| Traditional, Transformers eager, resident | 413.46 | 32.90 | Reference |
| Independent executor, flat, one full matrix at a time | 80.11 | 11.14 | All logits bit-identical |
| Independent executor, faces, one full matrix at a time | 80.11 | 5.90 | All logits bit-identical |
| Flat, requested embedding rows and 2,048-row head windows | 37.62 | 11.92 | All tokens matched |
| Faces, same row windows | 37.62 | 8.45 | All tokens matched |
| Flat, direct GPU arithmetic, same row windows | 36.23 | 10.06 | All tokens matched |
| Faces, direct GPU arithmetic, same row windows | 36.23 | 9.89 | All tokens matched |

Six prompts, 24 greedy tokens per prompt, batch one, FP32. All 4,608,000 scores matched the resident bit patterns in each whole-matrix arm. Smaller GEMMs and direct reductions changed rounding. Maximum absolute error versus resident was 9.06e-6 for windowed dense GEMMs and 8.11e-6 for direct face arithmetic. The fresh resident scores also matched the original published reference bit for bit. In a second direct-kernel run in reverse order, each arm repeated its own logits hash and its 36.23 MiB peak. Speeds were 9.44 tokens/s flat and 8.19 faces. Timings include synchronized forward and argmax. They vary across runs. [Second flat run](results/repeat-flat-direct.json) · [Second face run](results/repeat-face-direct.json)

## What runs on its own

The earlier experiment placed addresses on faces and loaded weights from a conventional binary file. Here a tile contains four cubes with four layers of 4×4 cells. A leaf contains the **actual 32-bit weight value**. Two nested layer lists give its position. The compact file layout keeps the leaves in face-cell order and represents occupancy and layer tags with bits or, for a full tile tensor, implicit positions. It preserves every bit, including negative zero. The converter verified all 313 tensors against the pinned source and compared 10,016 sampled cells to the original Siybox encoder. Face weight files occupied 399,316,664 bytes, flat weight files 399,309,440 bytes, excluding manifests and tokenizer settings. [Packer](face_store.py) · [Packing receipt](results/pack-result.json)

The standalone program opens the face directory and its bundled tokenizer. The executor implements the checkpoint's normalization, rotary positions, attention, KV cache, learned top-two expert gate, expert layers and output head. PyTorch supplies tensor operations and Triton supplies the direct kernels. Inference uses no Transformers model, router or original weight file. It preserves the checkpoint's learned routing rule. The experiment does not exercise advance weight selection, online learning or a change to the trained architecture. The native three-rule language-model comparison uses its own architecture and has no expert gate. [Executor](executor.py) · [Direct GPU operations](direct_ops.py) · [Native model and matched controls](native/README.md)

The whole-matrix path rebuilds each requested matrix from face leaves. The direct path maps the input row and column to the physical leaf inside the GPU kernel, loads the scalar and uses it in the multiply. Across the six prompts, the direct path made 19,125 fully occupied face reads and 3,024 compact face reads. One independent check blocked dense weight reconstruction, increased a single output-head leaf by 1 in a copy-on-write mapping, and reran a seventh prompt. Exactly the addressed score changed out of 32,000. Restoring the leaf restored every score bit for bit. [Intervention](check_direct.py) · [Observed result](results/causal-check.json)

Peak allocation covers live PyTorch GPU allocations during single-token decode. The report also records peak reserved memory, prefill, setup, host process peak RSS and bytes requested. Direct face decoding reserved up to 56 MiB during decode. Face direct process peak host RSS was about 1.85 GiB including the runtime and mapped pages, versus about 1.62 GiB for resident Transformers. The direct face run addressed about 20.8 GB of encoded weights across prompt processing and generation. All eight experts per layer remained addressable. The native gate used two per token per layer. The complete pool stays on the host.

## Run it

Use Linux or WSL with CUDA. The published runs used Python 3.12.3, PyTorch 2.12.1+cu130, Transformers 5.12.1 for the resident control, NumPy, tokenizers, safetensors for conversion, and Triton for the direct GPU path. The [standalone data release](https://github.com/MidKnight-Rising/siybox-weight-pool/releases/tag/standalone-2026-11-23) contains the complete face store, the flat store, both tokenizer copies, raw run-A logits, and SHA-256 checksums. Use an empty data directory and run the Python commands from the repository root.

```bash
mkdir -p standalone-data
gh release download standalone-2026-11-23 -R MidKnight-Rising/siybox-weight-pool --dir standalone-data
(cd standalone-data && sha256sum -c SHA256SUMS && tar xzf standalone-data.tar.gz && sha256sum -c CONTENTS.sha256)
DATA="$PWD/standalone-data"
python standalone/generate.py "$DATA/standalone-faces" "A copper kettle sat on the stove" --tokens 6 --direct
python standalone/check_direct.py "$DATA/standalone-faces"
```

To regenerate the stores or benchmark the traditional runtime, set `MODEL` to the pinned original checkpoint's `model/` directory from the [first release](https://github.com/MidKnight-Rising/siybox-weight-pool/releases/tag/experiment-2026-11-23). The packer writes two fresh directories. The two stores in the standalone download are ready to use immediately.

```bash
python standalone/face_store.py pack --model "$MODEL" --faces rebuilt-faces --flat rebuilt-flat
OUT="$PWD/standalone-replay"
python standalone/benchmark.py resident --root "$MODEL" --out "$OUT"
for mode in flat face; do
  if [ "$mode" = face ]; then root="$DATA/standalone-faces"; else root="$DATA/standalone-flat"; fi
  python standalone/benchmark.py "$mode" --root "$root" --out "$OUT"
  python standalone/benchmark.py "$mode" --root "$root" --rows 2048 --out "$OUT"
  python standalone/benchmark.py "$mode" --root "$root" --rows 2048 --direct --out "$OUT"
done
python standalone/benchmark.py compare --out "$OUT"
```
