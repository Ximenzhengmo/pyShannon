[English Docs](./docs/en/README-en.md) | [中文文档](./docs/zh/README-zh.md)

# pyShannon

`pyShannon` is a Python toolkit for discrete information theory and stochastic process modeling.
`pyShannon` 是一个面向离散信息论与随机过程建模的 Python 工具库。

This project has a strong vibe coding influence, with AI contributions in both code and documentation.
目前本项目有 vibe coding 的大量参与，代码和文档中 AI 味十足

## Current Coverage

The current implementation includes:

1. Information metrics and entropy APIs (`I(x)`, `H(X)`, `H(X|Y)`, `I(X;Y)`)
2. `MemoryLessChain` and higher-order `MarkovChain`
3. Unified source wrapper `Source` (memoryless / Markov)
4. Unified channel wrapper `Channel` (memoryless / Markov)
5. Huffman coding/decoding (`HuffmanCodec`)
6. Integrated demos:
   - Source -> Huffman encode/decode
   - Source -> Huffman -> Channel -> Huffman

## Requirements

```text
Python >= 3.11
NumPy >= 1.25
```

## Run Demo

```bash
python example.py
```

`example.py` demonstrates all currently implemented features end-to-end.

## Documentation

For full bilingual documentation (including LaTeX formulas that explain the theory behind each demo section):

1. English: [docs/en/README-en.md](./docs/en/README-en.md)
2. 中文: [docs/zh/README-zh.md](./docs/zh/README-zh.md)

## Project Structure

1. `probability.py`: information metrics and entropy utilities
2. `chain.py`: `MemoryLessChain`, `MarkovChain`
3. `source.py`: unified source factory/wrapper
4. `channel.py`: unified channel factory/wrapper
5. `codec.py`: Huffman tree, encoding, decoding
6. `example.py`: complete runnable examples
