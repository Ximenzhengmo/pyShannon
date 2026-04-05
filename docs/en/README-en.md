[English](./README-en.md) | [中文](../zh/README-zh.md)

# pyShannon (English Docs)

`pyShannon` is a Python toolkit for discrete information theory and stochastic-process modeling. The project is no longer just a collection of scripts: it now includes

1. A reusable Python computation core
2. CLI examples and a problem-solving script
3. A local Flask-based web calculator

## Current Coverage

The current version supports:

1. Information metrics and entropy calculations
2. Memoryless chain and higher-order Markov chain
3. Memoryless source and Markov source
4. Memoryless channel and Markov channel
5. Huffman encoding and decoding
6. Two integrated pipelines (source-codec, source-codec-channel-codec)
7. An interactive web calculator for information-theoretic tasks

## Requirements

```text
Python >= 3.11
NumPy >= 1.25
Flask >= 3.1
```

If you want to match the current local development setup, use the conda environment `pythontest`.

## Python-side Usage

### 1) Run the full example script

```bash
python example.py
```

### 2) Run the worked problem solver

```bash
python solve.py
```

### 3) Public probability API

The public entropy function exported by `probability.py` is now:

```python
from probability import entropy
```

instead of the older `Entropy(...)` naming.

Common APIs include:

1. `self_information(p)`
2. `entropy(p)`
3. `conditional_entropy(p_cond, p_joint)`
4. `mean_mutual_information(pX, pY, pXY)`
5. `conditional_distribution(p_joint, p_condition)`
6. `safe_divide(numerator, denominator)`

`conditional_distribution(...)` and `safe_divide(...)` are especially useful for building conditional distributions safely without producing `nan` on zero-probability branches.

## Web Calculator Guide

The web app lives in:

1. `web/app.py`
2. `web/calculator.py`
3. `web/templates/index.html`
4. `web/static/app.js`
5. `web/static/styles.css`

### Launch

The recommended launch command in the current environment is:

```bash
conda run -n pythontest python web/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Main Web Features

#### 1) Base and derived variables

You can:

1. Add base random variables such as `X`, `Y`, `Z`
2. Assign LaTeX names to them
3. Define state sets like `0,1` or `A,B,C`
4. Add derived variables such as `Z = X*Y`

#### 2) Joint probability input

The page auto-builds a joint probability table from the base variables, and you only need to fill in the probabilities.

Supported helpers:

1. Uniform fill
2. Normalize current table
3. Normalize automatically before submission
4. One-click loading of the exam-style sample

#### 3) LaTeX rendering

The page supports:

1. Variable preview in LaTeX
2. Formula rendering in result cards
3. Formula rendering in recursive explanation panels

#### 4) Formula focus and recursive explanation

In the result area you can:

1. Click any formula directly
2. Type formulas such as `H(X|Y)` or `I(X;Y|Z)`
3. Focus one or more formulas and recursively expand every dependency needed to compute them

For example,

$$
I(X;Y\mid Z)=H(X\mid Z)-H(X\mid Y,Z)
$$

The page will automatically show:

1. `I(X;Y|Z)`
2. `H(X|Z)`
3. `H(X|YZ)`
4. and any lower-level dependencies needed by those expressions

#### 5) Generated Python code

The page also produces a Python code snippet showing how to compute the selected formulas from the project root using this library. It includes:

1. Building the joint-distribution tensor
2. Building marginals when needed
3. Calling `entropy(...)` or combining intermediate formulas
4. Automatically appending `print(...)` lines for the focused formulas

A one-click copy button is also provided.

#### 6) 2D probability-table visualization

The result page includes a 2D probability-table view. You can edit which variables appear on the row axis and column axis.

For example, you can choose:

1. Row axis: `X`
2. Column axis: `YZ`

which gives a table view of a distribution like

$$
P(X, YZ)
$$

or any other valid disjoint row/column variable grouping.

#### 7) Internationalization

The web UI defaults to Chinese and also supports a one-click switch to English.

## Theory Snapshot for Demo Sections (with LaTeX)

### 1) Information Metrics and Entropy

Self-information:

$$
I(x)=-\log_2 p(x)
$$

Entropy:

$$
H(X)=-\sum_x p(x)\log_2 p(x)
$$

Conditional entropy:

$$
H(X|Y)=-\sum_{x,y} p(x,y)\log_2 p(x|y)
$$

Mutual information:

$$
I(X;Y)=\sum_{x,y} p(x,y)\log_2\frac{p(x,y)}{p(x)p(y)}
$$

### 2) Memoryless Chain (`MemoryLessChain`)

For i.i.d. symbols:

$$
P(x_{1:k})=\prod_{t=1}^{k} P(x_t)
$$

### 3) m-th Order Markov Chain (`MarkovChain`)

$$
P(x_{1:k})=P(x_{1:m})\prod_{t=m+1}^{k}P\bigl(x_t\mid x_{t-m:t-1}\bigr)
$$

### 4) Two Source Types (`Source`)

- Memoryless source: sampling from a fixed distribution.
- Markov source: context-driven sampling:

$$
P(s_t\mid s_{1:t-1})=P(s_t\mid s_{t-m:t-1})
$$

### 5) Two Channel Types (`Channel`)

#### 5.1 Discrete Memoryless Channel (DMC)

$$
P(y_{1:n}\mid x_{1:n})=\prod_{t=1}^{n}P(y_t\mid x_t)
$$

Output and joint distributions:

$$
p_Y(y)=\sum_x p_X(x)P(y\mid x),\quad p_{XY}(x,y)=p_X(x)P(y\mid x)
$$

#### 5.2 Discrete Markov Channel

$$
P(y_t\mid x_{1:t})=P\bigl(y_t\mid x_{t-m+1:t}\bigr)
$$

Hence:

$$
P(y_{1:n}\mid x_{1:n})=\prod_{t=1}^{n}P\bigl(y_t\mid x_{t-m+1:t}\bigr)
$$

### 6) Huffman Coding/Decoding

Given symbol probabilities $\{p_i\}$ and code lengths $l_i$, average code length:

$$
\bar L=\sum_i p_i l_i
$$

with the classic bound:

$$
H(X)\le \bar L < H(X)+1
$$

### 7) Integrated Example A: Source -> Huffman Encode/Decode

$$
b=E(s_{1:n}),\quad \hat s_{1:n}=D(b)
$$

Under error-free transmission/storage:

$$
\hat s_{1:n}=s_{1:n}
$$

### 8) Integrated Example B: Source -> Huffman -> Channel -> Huffman

$$
b=E(s_{1:n}),\quad r=C(b),\quad \hat s_{1:n}=D(r)
$$

If the channel is error-free:

$$
r=b\Rightarrow \hat s_{1:n}=s_{1:n}
$$

For a lossy channel (e.g. bit-flip probability $\varepsilon$, bit length $L$):

$$
P(r=b)=(1-\varepsilon)^L
$$

So exact recovery is no longer guaranteed, and decoding may fail because of variable-length code desynchronization.

## Module Index

1. `probability.py`: information metrics, entropy, conditional entropy, mutual information, and safe conditional-distribution helpers.
2. `chain.py`: `MemoryLessChain` and `MarkovChain`.
3. `source.py`: unified `Source` wrapper for memoryless/Markov sources.
4. `channel.py`: unified `Channel` wrapper for memoryless/Markov channels.
5. `codec.py`: `HuffmanCodec`, `huffman_encoding`, `huffman_encode`, `huffman_decode`.
6. `web/`: the local Flask calculator application.

## Note

These docs are now aligned with `example.py`, `solve.py`, and the web calculator. If the project grows further, it is a good idea to keep the Python examples and web usage notes updated together.
