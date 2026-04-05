[English Docs](./docs/en/README-en.md) | [中文文档](./docs/zh/README-zh.md)

# pyShannon

`pyShannon` is a local information-theory toolkit centered on discrete distributions, Markov processes, channels, and source/channel coding.
`pyShannon` 是一个面向离散信息论建模与仿真的本地 Python 工具库。

## Scope

Current functionality includes:

1. Probability-side information metrics: `self_information`, `entropy`, `conditional_entropy`, `mean_mutual_information`
2. Discrete stochastic models: `MemoryLessChain`, `MarkovChain`
3. Unified source wrapper: `Source`
4. Unified channel wrapper: `Channel`
5. Huffman coding/decoding: `HuffmanCodec`
6. Tolerant Huffman decoding for noisy bitstreams: `HuffmanCodec.tolerant_decode(...)`
7. Example scripts, worked problem solver, and experiment rewrites under `exp/`
8. A local Flask-based web calculator under `web/`

## Interface Notes

The public entropy API in `probability.py` is:

```python
from probability import entropy
```

Use `entropy(...)` instead of the older `Entropy(...)` name.

For conditional distributions with zero-probability branches, prefer:

```python
from probability import conditional_distribution, safe_divide
```

## Requirements

```text
Python >= 3.11
NumPy >= 1.25
Flask >= 3.1
```

In the current local setup, the recommended conda environment is `pythontest`.

## Quick Start

Run the end-to-end example script:

```bash
python example.py
```

Run the worked problem solver:

```bash
python solve.py
```

Run the local web calculator:

```bash
conda run -n pythontest python web/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Experiments

Rewritten experiment scripts are under `exp/`:

1. `experiment1_stationary_sources.py`
2. `experiment2_cascade_channels.py`
3. `experiment3_huffman_image_compression.py`
4. `experiment4_reliability_vs_efficiency.py`

These scripts are written against the current library interfaces rather than the older standalone implementations in the reports.

## Documentation

Detailed bilingual documentation is in:

1. English: [docs/en/README-en.md](./docs/en/README-en.md)
2. 中文: [docs/zh/README-zh.md](./docs/zh/README-zh.md)

Those documents include:

1. Theory notes with LaTeX formulas
2. Python-side usage notes
3. Web calculator usage instructions
4. Current module overview

## Project Layout

1. `probability.py`: information metrics and safe conditional-distribution helpers
2. `chain.py`: `MemoryLessChain`, `MarkovChain`
3. `source.py`: unified source wrapper
4. `channel.py`: unified channel wrapper
5. `codec.py`: Huffman tree, encoding, strict decoding, tolerant decoding
6. `example.py`: end-to-end examples
7. `solve.py`: worked probability exercise
8. `web/`: Flask app and front-end assets
9. `exp/`: experiment scripts and reports
