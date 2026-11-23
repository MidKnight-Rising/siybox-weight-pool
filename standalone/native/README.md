# Original native Siybox versus learned attention

We trained the historical three-rule Siybox language model (both, neither and near) and its single-head learned causal attention control on the same token stream. Both used FP32, width 120, depth 3, batch 8, context 128, seed 1337 and 400 updates on Tiny Shakespeare (409,600 training targets). These are small word-level models. Neither uses an expert router. Each inference measurement loaded its trained state in a new process on the same RTX 2070 SUPER.

| Model and raw result | Weight storage | At rest after load (MiB) | Peak decode allocated (MiB) | Decode tokens/s | Validation loss |
|---|---|---:|---:|---:|---:|
| [Native Siybox](final-agreement-resident.json) | Fully resident | 9.796 | 22.591 | 697.2 | 4.6986 |
| [Learned attention](final-attention-resident.json) | Fully resident | 9.565 | 22.359 | 702.1 | 4.7749 |
| [Native Siybox](final-agreement-resident-shared.json) | Resident, identical fixed tables shared | 9.296 | 22.091 | 340.5 | 4.6986 |
| [Learned attention](final-attention-resident-shared.json) | Resident, identical fixed tables shared | 9.440 | 22.234 | 908.5 | 4.7749 |
| [Native Siybox](final-agreement-face.json) | Layer streaming, scalar faces | 0.750 | 16.562 | 96.2 | 4.6986 |
| [Learned attention](final-attention-face.json) | Layer streaming, scalar faces | 0.188 | 16.000 | 94.4 | 4.7749 |
| [Native Siybox](final-agreement-flat.json) | Layer streaming, flat arrays | 0.750 | 16.562 | 125.4 | 4.6986 |
| [Learned attention](final-attention-flat.json) | Layer streaming, flat arrays | 0.188 | 16.000 | 106.4 | 4.7749 |

The historical resident implementation used 0.231 MiB more peak memory for Siybox than for attention. It has fewer learned parameters (2,370,840 versus 2,457,240), but stores three fixed rule tables and a causal mask per block. Attention stores one causal mask per block. In a follow-up identified after inspecting that overhead, we shared identical fixed buffers across blocks in **both** models. The resident comparison then favored Siybox by 0.144 MiB (0.65%). The learned parameters stayed on the GPU. This narrow difference applies to this width and context.

In the streaming runs we held fixed buffers on the GPU and brought in every learned parameter of each leaf module for its original PyTorch operation. We released its weights after the operation. The flat and face stores gave the same peak within each architecture. All four storage configurations for each trained checkpoint produced the same SHA-256 over **231,424,000 FP32 validation logits** (28,928 targets × 8,000 words). Siybox predicted 7,003 targets correctly and attention predicted 6,703. Comparing losses across architectures describes one seed. Timing is the median of three warm 64-token generations at batch one. Both models recompute the full 128-token window per step.

## Reproduction

The JSON files linked above are public. This supplemental harness uses the original native model definitions from the private historical Siybox repository, commit `824c88ec7f5381599f6e7dd3a4caf840bf2611d2`, directory `original-language-source`. It verifies the source hashes before training or loading the graph. The standalone TinyMoE experiment one directory up can be reproduced entirely from public files. The older [agreement](agreement.json) and [attention](attention.json) receipts measured inference in the training process and are preserved for provenance. The table uses fresh-process measurements.

From this directory with CUDA PyTorch, set `SOURCE_DIR` to the original definitions and use the following commands. The word vocabulary is built on the training split. The final 10% of the text provides validation windows.

```bash
curl -fL https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt -o input.txt
for arm in agreement attention; do
  python benchmark.py train "$arm" --source "$SOURCE_DIR" --corpus input.txt --checkpoint "$arm.pt"
  for layout in resident resident-shared; do
    python benchmark.py measure "$arm" --source "$SOURCE_DIR" --corpus input.txt --checkpoint "$arm.pt" --storage "$layout" --out "$arm-$layout.json"
  done
  for layout in face flat; do
    python storage.py "$layout" --checkpoint "$arm.pt" --root "$arm-$layout"
    python benchmark.py measure "$arm" --source "$SOURCE_DIR" --corpus input.txt --root "$arm-$layout" --storage "$layout" --out "$arm-$layout.json"
  done
done
```
