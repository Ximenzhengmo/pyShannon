[English](./README-en.md) | [中文](../zh/README-zh.md)

# pyShannon (English Docs)

`pyShannon` is a Python toolkit for discrete information theory and stochastic process modeling. The current demo script [example.py](../../example.py) covers:

1. Information metrics and entropy variants
2. Memoryless chain and higher-order Markov chain
3. Memoryless source and Markov source
4. Memoryless channel and Markov channel
5. Huffman encoding and decoding
6. Two integrated pipelines (source-codec, source-codec-channel-codec)

## Requirements

```text
Python >= 3.11
NumPy >= 1.25
```

## Run Demo

```bash
python example.py
```

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

`prob_topk(k)` returns the joint distribution, and `prob_k([...])` gives selected marginals/joints.

### 3) m-th Order Markov Chain (`MarkovChain`)

$$
P(x_{1:k})=P(x_{1:m})\prod_{t=m+1}^{k}P\bigl(x_t\mid x_{t-m:t-1}\bigr)
$$

`setup(...)` sets the initial distribution term (e.g. $P(x_{1:m})$).

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

Let the source sequence be $s_{1:n}$ and codec be $(E,D)$:

$$
b=E(s_{1:n}),\quad \hat s_{1:n}=D(b)
$$

Under error-free transmission/storage:

$$
\hat s_{1:n}=s_{1:n}
$$

### 8) Integrated Example B: Source -> Huffman -> Channel -> Huffman

With channel operator $C$:

$$
b=E(s_{1:n}),\quad r=C(b),\quad \hat s_{1:n}=D(r)
$$

If the channel is error-free (as in the deterministic demo channel):

$$
r=b\Rightarrow \hat s_{1:n}=s_{1:n}
$$

For a lossy channel (e.g. bit flip probability $\varepsilon$, bit length $L$):

$$
P(r=b)=(1-\varepsilon)^L
$$

So exact recovery is no longer guaranteed; decoding may fail due to variable-length code desynchronization.

## Module Index

1. `probability.py`: information metrics and entropy APIs.
2. `chain.py`: `MemoryLessChain` and `MarkovChain`.
3. `source.py`: unified `Source` wrapper for memoryless/Markov sources.
4. `channel.py`: unified `Channel` wrapper for memoryless/Markov channels.
5. `codec.py`: `HuffmanCodec`, `huffman_encoding`, `huffman_encode/decode`.

## Note

This documentation is aligned with `example.py`. If new capabilities are added, update both the code snippets and formulas here.
