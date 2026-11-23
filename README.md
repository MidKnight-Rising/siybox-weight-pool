# Siybox weight pool

Can one model keep all its weights available while loading only the experts it needs on the GPU? On an RTX 2070 SUPER, an open 99.8M-parameter TinyMoE checkpoint ran from a complete Siybox pool with **64.5% lower peak PyTorch-allocated memory during decoding**. All 144 generated tokens and all 4,608,000 next-token logits matched the fully resident model bit for bit. Decoding slowed from 32.8 to 23.0 tokens per second. [Run receipt](RESULT.json)

| How the same checkpoint ran | Peak allocated during decode (MiB) | Decode tokens/s | Output versus resident |
|---|---:|---:|---|
| Fully resident on the GPU | 413.5 | 32.8 | Reference |
| Experts loaded on demand through a flat index | 146.7 | 20.9 | Bit-identical |
| Experts loaded on demand through Siybox cube faces | 146.7 | 23.0 | Bit-identical |

The flat index reached the same memory peak. Selective expert loading accounts for the measured savings. The cube run shows that Siybox can serve the real weights in the middle of generation and preserve every checked output. Throughput values are from the packaged second run on one GPU. [First and second runs](https://github.com/MidKnight-Rising/siybox-weight-pool/releases/tag/experiment-2026-11-23)

## Standalone follow-up

The original run below used faces as an index into a weight file. The follow-up stores the FP32 values themselves on nested faces and runs the entire checkpoint with an independent executor. Direct GPU arithmetic on face leaves used **36.23 MiB** peak during decode versus **413.46 MiB** resident. The matched flat kernel also used **36.23 MiB**. All 144 tokens matched in both streaming arms. The follow-up also measures the original native Siybox language model against learned attention. [Standalone results, code and reproduction](standalone/README.md)

## Where the weights go

The packer verifies 313 source tensors and writes all 99,817,344 FP32 parameters to one host-side file (399,269,376 bytes). The Siybox face entries point to byte ranges in that file. Shared weights stay on the GPU. At each of ten layers, the model's own router chooses two of eight experts for a single decode token. The runner looks up the layer, expert and projection on the faces, transfers those weights to the GPU, applies them, and releases them. The same expert can be fetched again on a later token. [Encoder](siybox.py) Ã‚Â· [Packer and inference code](weight_pool.py) Ã‚Â· [Face manifest and tensor hashes](POOL_MANIFEST.json)

Every single-token decode step fetched 20 of the 80 expert groups. Across the six prompts and their continuations, the run exercised 78 groups. Packing checked all 80. Each prompt produced 24 tokens by greedy generation. The first token comes from processing the prompt, leaving 138 single-token steps for the decode timing. Memory is peak live tensor allocation reported by PyTorch on this workload. Whole-device usage, host RAM, larger checkpoints, dense models and anticipatory weight selection remain open questions.

## Evidence for the earlier Siybox results

- **Word similarity.** The depth-two Siybox score on WordSim-353 was **0.674** against **0.609** for GloVe-300 on the same 353 pairs. GloVe led on the other four datasets. A same-corpus word2vec skip-gram control scored 0.6729 on WordSim-353. [All five scores and original output](evidence/README.md#word-similarity)
- **Synthetic copying.** On a marked-copy task, Siybox-derived directed reading averaged **98.04%** in-distribution token accuracy at 3,000 optimizer steps. Three-head attention reached **2.80%** at the same budget and **97.45%** after continuation to 12,000 steps. Directed reading led on longer gaps. The extended attention model led on dense collisions. [Protocols, per-seed results and comparisons](evidence/README.md#marked-copy-task)
- **Natural-language modeling.** In a separate three-seed small-model test, the agreement model averaged **4.8007** validation loss and attention averaged **4.7740**. Attention had the lower mean. These runs support calling the observed losses close. Establishing equivalence would require a defined margin and more evidence. [Per-seed validation losses](evidence/README.md#natural-language-modeling)

## Complete experiment data

The [experiment release](https://github.com/MidKnight-Rising/siybox-weight-pool/releases/tag/experiment-2026-11-23) contains the original open checkpoint and its settings, both generated pools, both experiment runs with raw logits, and the upstream model card. The archive uses standard TAR hard links for identical files. Extract it into one directory to keep all paths and check the included artifact hashes.

```bash
mkdir -p data
gh release download experiment-2026-11-23 --repo MidKnight-Rising/siybox-weight-pool --dir data
(cd data && sha256sum -c SHA256SUMS && tar -xzf siybox-weight-pool-experiment.tar.gz && sha256sum -c ARTIFACTS.sha256)
```

The archive's `model/` is the pinned [TinyMoE checkpoint](https://huggingface.co/FlameF0X/TinyMoE-100m-2x8-retrained/tree/0c1d097c964b949bc0092fa8e99c88c325421167), identified as MIT by its author. `pool2/` is the verified pool with packaged settings. `run2/` is the corresponding benchmark. `pool/` and `run1/` preserve the first run. The bundled `siybox.py` is the byte-identical encoder used to create the faces.

## Reproduce

The original run used Python 3.12.3, PyTorch `2.12.1+cu130`, Transformers `5.12.1`, safetensors, and CUDA. Run these commands from the repository with `DATA_DIR` pointing to the extracted archive. Set the three paths below to fresh output directories and the directory containing the extracted data. The data directory should contain `ARTIFACTS.sha256`.

```bash
DATA_DIR="$PWD/data"       # extract the archive here
RUN_DIR="$PWD/replay"      # new benchmark output
NEW_POOL_DIR="$PWD/repacked"

python weight_pool.py resident --model "$DATA_DIR/model" --pool "$DATA_DIR/pool2" --out "$RUN_DIR"
python weight_pool.py flat --pool "$DATA_DIR/pool2" --out "$RUN_DIR"
python weight_pool.py cube --pool "$DATA_DIR/pool2" --out "$RUN_DIR"
python weight_pool.py compare --out "$RUN_DIR"
```

To regenerate a standalone pool from the original checkpoint, use the bundled encoder.

```bash
python weight_pool.py pack --model "$DATA_DIR/model" --pool "$NEW_POOL_DIR" \
  --siybox siybox.py --out "$RUN_DIR"
```

The resident and pooled processes preserve the original routing and expert arithmetic. The comparison checks the bit patterns of the float32 logits. Timings use synchronized CUDA forward and argmax intervals for a batch of one. PyTorch allocation measures live tensor allocations on the GPU.
