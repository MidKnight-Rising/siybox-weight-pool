# siybox-weight-pool

One open-weight model runs from a complete Siybox layer-list weight pool. The model's existing router supplies expert addresses. Siybox follows the recorded face and depth to load the requested tensor spans for ordinary model computation.

## Results

- Packed all **99,817,344 parameters** from **313 source tensors** and verified the original bytes. All **80 expert groups** were addressable.
- The fully resident, flat-index offload, and Siybox arms agreed on **144 generated tokens** and **4,608,000 float32 logits** bit for bit across six prompts.
- On an RTX 2070 SUPER, peak PyTorch-allocated VRAM during decoding was **413.5 MiB** fully resident and **146.7 MiB** in both pooled arms, a **64.5%** reduction. Decode rates were **32.8**, **20.9**, and **23.0 tokens/s** for resident, flat, and Siybox, respectively, in the packaged run.
- Each decode step fetched **20 of 80 expert groups**. Across the full run, including prompt processing, the route exercised **78** groups.
- The flat-index control achieved the same memory result. The savings measure selective expert loading. The Siybox arm demonstrates that the cube-face representation can serve the actual weights losslessly during generation.

[Benchmark receipt](RESULT.json) · [Weight pool manifest](POOL_MANIFEST.json)

## Complete experiment data

The [experiment release](https://github.com/MidKnight-Rising/siybox-weight-pool/releases/tag/experiment-2026-11-23) contains the original open checkpoint and its settings, both generated pools, both experiment runs with raw logits, and the upstream model card. The archive uses standard TAR hard links for identical files. Extract it into one directory to keep all paths and check the included artifact hashes.

```bash
sha256sum -c SHA256SUMS
tar -xzf siybox-weight-pool-experiment.tar.gz
sha256sum -c ARTIFACTS.sha256
```

The archive's `model/` is the pinned [TinyMoE checkpoint](https://huggingface.co/FlameF0X/TinyMoE-100m-2x8-retrained/tree/0c1d097c964b949bc0092fa8e99c88c325421167), identified as MIT by its author. `pool2/` is the verified pool with packaged settings. `run2/` is the corresponding benchmark. `pool/` and `run1/` preserve the first run. The bundled `siybox.py` is the byte-identical encoder used to create the faces.

## Reproduce

The original run used Python 3.12.3, PyTorch `2.12.1+cu130`, Transformers `5.12.1`, safetensors, and CUDA. Run these commands from the repository with `DATA_DIR` pointing to the extracted archive. The output directory can be any writable directory.

```bash
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
