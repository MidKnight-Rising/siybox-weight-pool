# Earlier Siybox benchmarks

These experiments explain why we tried using a cube-face index for model weights. Each measures a different use of Siybox. The TinyMoE weight-pool experiment uses the checkpoint's existing attention and router. [Weight-pool measurements](../RESULT.json)

## Word similarity

How closely do each method's scores track human judgments of word pairs? Both methods were graded by Spearman rank correlation on the same pairs they could score. Higher is better. The Siybox column uses the depth-two recursive layer-list scorer with fold size 10, built on 84,088,239 WikiText-103 tokens. The comparison used the pretrained 300-dimensional GloVe Wiki + Gigaword 6B vectors.

| Benchmark | Pairs graded | Siybox | GloVe-300 |
|---|---:|---:|---:|
| WordSim-353 | 353 / 353 | **0.674** | 0.609 |
| SimLex-999 | 999 / 999 | 0.241 | **0.371** |
| MEN | 3,000 / 3,000 | 0.624 | **0.737** |
| RG-65 | 65 / 65 | 0.651 | **0.766** |
| Rare Words | 1,702 / 2,034 | 0.138 | **0.408** |

The replay log includes all four fold depths. The input record gives the configuration, data filenames, SHA-256 hashes and library versions used to obtain the scores. [Original run output](word-similarity/stdout.txt) · [Input record](word-similarity/inputs.json) · [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) · [WordSim-353](https://gabrilovich.com/resources/data/wordsim353/wordsim353.html)

A later reference trained 300-dimensional word2vec on the same WikiText-103 paragraphs. Skip-gram reached 0.6729 on the same 353 WordSim pairs, close to the depth-two Siybox score. CBOW reached 0.6516. [Same-corpus baseline results](word-similarity/matched_baseline_results.json)

## Marked copy task

The model sees a marked sequence of symbols, a gap containing distractors and a second start marker. Its task is to copy the source sequence. The original comparison held width (192), depth (4), training stream per seed and budget fixed. The directed reader and learned three-head attention had 1,605,888 and 1,900,800 parameters, respectively. Both were evaluated on the same 512 held-out sequences per condition. The follow-up continued the three attention checkpoints for another 9,000 steps and evaluated the final checkpoints on those frozen sets. Optimizer steps measure training budget here.

| Held-out metric (three-seed mean) | Directed reading, 3k steps | Attention, 3k steps | Attention, 12k steps |
|---|---:|---:|---:|
| In-distribution token accuracy | 98.04% | 2.80% | 97.45% |
| In-distribution whole sequences copied exactly | 49.35% | 0.00% | 37.96% |
| Unseen lengths, token accuracy | 98.19% | 2.69% | 97.50% |
| Longer patterns, token accuracy | 95.10% | 1.84% | 92.14% |
| Longer gaps, token accuracy | 96.45% | 2.80% | 84.88% |
| Unique source symbols, token accuracy | 100.00% | 1.94% | 99.46% |
| Dense collisions, token accuracy | 22.36% | 6.07% | 74.78% |

Exact copy counts a sequence only when every target token is correct. In the dense-collision condition, both directed reading and the extended attention model scored zero whole-sequence exact copies. The per-seed files include integer hit counts and denominators. [Original protocol](copy-task/original/PROTOCOL.json) · [Original summary and seed results](copy-task/original) · [Continuation protocol](copy-task/extension/PROTOCOL.json) · [Continuation summary and seed results](copy-task/extension)

## Natural-language modeling

A separate experiment compared a three-rule agreement mixer with learned attention on next-token prediction. Both used width 120, depth 3, context 128, batch size 16 and 3,000 training steps. The table reports the best observed validation loss for each run in nats per token. Lower is better.

| Training seed | Three-rule agreement | Attention |
|---|---:|---:|
| 1337 | 4.7651 | 4.7964 |
| 1438 | 4.8324 | 4.7708 |
| 1539 | 4.8045 | 4.7548 |
| Mean | **4.8007** | **4.7740** |

The agreement model led on the first seed. Attention led on the other two. The mean difference was 0.0267 nats per token in attention's favor. The sample is three paired runs. [Full-precision losses for all three seeds](language-model/seed_confirm.json)
