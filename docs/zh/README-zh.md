[English](../en/README-en.md) | [中文](./README-zh.md)

# pyShannon（中文文档）

`pyShannon` 是一个面向离散信息论与随机过程建模的 Python 工具库，当前示例脚本 [example.py](../../example.py) 覆盖了：

1. 各类信息量与熵计算
2. 无记忆链与高阶马尔可夫链
3. 无记忆信源与马尔可夫信源
4. 无记忆信道与马尔可夫信道
5. 哈夫曼编码与解码
6. 两类联合流程（信源-编解码、信源-编码-信道-解码）

## 环境

```text
Python >= 3.11
NumPy >= 1.25
```

## 运行示例

```bash
python example.py
```

## 示例理论速览（含公式）

### 1) 信息量与熵

单事件自信息：

$$
I(x)=-\log_2 p(x)
$$

熵：

$$
H(X)=-\sum_x p(x)\log_2 p(x)
$$

条件熵：

$$
H(X|Y)=-\sum_{x,y} p(x,y)\log_2 p(x|y)
$$

互信息：

$$
I(X;Y)=\sum_{x,y} p(x,y)\log_2\frac{p(x,y)}{p(x)p(y)}
$$

### 2) 无记忆链（MemoryLessChain）

若符号独立同分布，则：

$$
P(x_{1:k})=\prod_{t=1}^{k} P(x_t)
$$

对应代码里 `prob_topk(k)` 给出联合分布，`prob_k([...])` 给出选定时刻的边缘/联合分布。

### 3) m 阶马尔可夫链（MarkovChain）

$$
P(x_{1:k})=P(x_{1:m})\prod_{t=m+1}^{k}P\bigl(x_t\mid x_{t-m:t-1}\bigr)
$$

`setup(...)` 用于指定初始分布（例如 $P(x_{1:m})$）。

### 4) 两类信源（Source）

- 无记忆信源：按固定分布采样符号。
- 马尔可夫信源：按状态上下文采样，满足

$$
P(s_t\mid s_{1:t-1})=P(s_t\mid s_{t-m:t-1})
$$

### 5) 两类信道（Channel）

#### 5.1 离散无记忆信道（DMC）

$$
P(y_{1:n}\mid x_{1:n})=\prod_{t=1}^{n}P(y_t\mid x_t)
$$

输出分布与联合分布：

$$
p_Y(y)=\sum_x p_X(x)P(y\mid x),\quad p_{XY}(x,y)=p_X(x)P(y\mid x)
$$

#### 5.2 离散马尔可夫信道

$$
P(y_t\mid x_{1:t})=P\bigl(y_t\mid x_{t-m+1:t}\bigr)
$$

于是：

$$
P(y_{1:n}\mid x_{1:n})=\prod_{t=1}^{n}P\bigl(y_t\mid x_{t-m+1:t}\bigr)
$$

### 6) 哈夫曼编码与解码

对符号概率分布 $\{p_i\}$，哈夫曼码字长度为 $l_i$，平均码长：

$$
\bar L=\sum_i p_i l_i
$$

并满足经典界：

$$
H(X)\le \bar L < H(X)+1
$$

### 7) 联合示例 A：信源 -> 哈夫曼编解码

记源符号序列为 $s_{1:n}$，哈夫曼编码器/解码器分别为 $E,D$：

$$
b=E(s_{1:n}),\quad \hat s_{1:n}=D(b)
$$

在无误码条件下：

$$
\hat s_{1:n}=s_{1:n}
$$

### 8) 联合示例 B：信源 -> 哈夫曼编码 -> 信道 -> 哈夫曼解码

记信道算子为 $C$：

$$
b=E(s_{1:n}),\quad r=C(b),\quad \hat s_{1:n}=D(r)
$$

若信道无误码（示例中的确定性信道）：

$$
r=b\Rightarrow \hat s_{1:n}=s_{1:n}
$$

若是有损信道（如比特翻转概率 $\varepsilon$、比特长度 $L$），则

$$
P(r=b)=(1-\varepsilon)^L
$$

因此不再保证正确还原，甚至可能触发解码异常（变长码失步）。

## 模块索引

1. `probability.py`：信息量、熵、互信息等函数。
2. `chain.py`：`MemoryLessChain` 与 `MarkovChain`。
3. `source.py`：统一 `Source` 包装无记忆/马尔可夫信源。
4. `channel.py`：统一 `Channel` 包装无记忆/马尔可夫信道。
5. `codec.py`：`HuffmanCodec`、`huffman_encoding`、`huffman_encode/decode`。

## 备注

当前文档与 `example.py` 示例脚本保持对应关系；若你新增功能，建议同步更新本节公式与代码片段。
