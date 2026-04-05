[English](../en/README-en.md) | [中文](./README-zh.md)

# pyShannon（中文文档）

`pyShannon` 是一个面向离散信息论与随机过程建模的 Python 工具库。当前项目已经不只是一个“计算脚本集合”，而是同时包含：

1. Python 计算核心
2. 命令行示例与题解脚本
3. 本地 Flask 网页计算器

## 当前能力

当前版本已经覆盖：

1. 各类信息量与熵计算
2. 无记忆链与高阶马尔可夫链
3. 无记忆信源与马尔可夫信源
4. 无记忆信道与马尔可夫信道
5. 哈夫曼编码与解码
6. 两类联合流程（信源-编解码、信源-编码-信道-解码）
7. 网页端交互式信息论计算器

## 环境

```text
Python >= 3.11
NumPy >= 1.25
Flask >= 3.1
```

如果你使用本项目当前开发环境，推荐直接使用本地 conda 环境 `pythontest`。

## Python 侧使用

### 1) 运行完整示例

```bash
python example.py
```

### 2) 运行题目求解脚本

```bash
python solve.py
```

### 3) 概率模块接口

当前 `probability.py` 中公开使用的熵函数接口为：

```python
from probability import entropy
```

而不是旧版本里的 `Entropy(...)`。

常用函数包括：

1. `self_information(p)`
2. `entropy(p)`
3. `conditional_entropy(p_cond, p_joint)`
4. `mean_mutual_information(pX, pY, pXY)`
5. `conditional_distribution(p_joint, p_condition)`
6. `safe_divide(numerator, denominator)`

其中 `conditional_distribution(...)` 和 `safe_divide(...)` 用于安全构造条件分布，避免在零概率条件分支上出现 `nan`。

## 网页端使用说明

网页端代码位于：

1. `web/app.py`
2. `web/calculator.py`
3. `web/templates/index.html`
4. `web/static/app.js`
5. `web/static/styles.css`

### 启动方式

推荐在 `pythontest` 环境中启动：

```bash
conda run -n pythontest python web/app.py
```

启动后打开：

```text
http://127.0.0.1:5000
```

### 网页端主要功能

#### 1) 基础变量与派生变量

- 可添加基础随机变量，例如 `X`, `Y`, `Z`
- 可为变量填写 LaTeX 名称
- 可设置状态集合，例如 `0,1` 或 `A,B,C`
- 可添加派生变量，例如 `Z = X*Y`

#### 2) 联合概率表输入

网页会根据基础变量自动生成联合概率表，你只需填写各行的概率值。

支持：

1. 均匀填充
2. 当前表归一化
3. 提交前自动归一化
4. 一键加载题图示例

#### 3) LaTeX 公式显示

网页支持：

1. 变量 LaTeX 预览
2. 结果公式 LaTeX 渲染
3. 递归计算过程中的公式展示

#### 4) 结果筛选与递归展开

在“计算结果”区域可以：

1. 直接点击任意公式
2. 手动输入公式，如 `H(X|Y)`、`I(X;Y|Z)`
3. 聚焦某个或某些公式后，递归展示该公式依赖的其他公式

例如：

$$
I(X;Y\mid Z)=H(X\mid Z)-H(X\mid Y,Z)
$$

网页会自动同时展示：

1. `I(X;Y|Z)`
2. `H(X|Z)`
3. `H(X|YZ)`
4. 以及它们继续依赖的底层公式

#### 5) 代码计算过程展示

网页会同时生成一段“项目根目录中使用本库的 Python 计算代码”，内容包括：

1. 构造联合分布张量
2. 按需求边缘分布
3. 调用 `entropy(...)` 或基于已有结果组合公式
4. 在代码末尾自动 `print(...)` 当前聚焦的公式结果

并支持一键复制代码。

#### 6) 二维概率分布表可视化

结果页新增了二维表格视图，用于把联合分布按“行轴变量 vs 列轴变量”展示。

你可以自行选择：

1. 纵轴变量，例如 `X`
2. 横轴变量，例如 `YZ`

这样就能看到类似：

$$
P(X, YZ)
$$

或更一般的二元表格形式。

#### 7) 中英文国际化

网页默认显示中文，同时支持一键切换到英文界面。

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

### 3) m 阶马尔可夫链（MarkovChain）

$$
P(x_{1:k})=P(x_{1:m})\prod_{t=m+1}^{k}P\bigl(x_t\mid x_{t-m:t-1}\bigr)
$$

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

$$
b=E(s_{1:n}),\quad \hat s_{1:n}=D(b)
$$

在无误码条件下：

$$
\hat s_{1:n}=s_{1:n}
$$

### 8) 联合示例 B：信源 -> 哈夫曼编码 -> 信道 -> 哈夫曼解码

$$
b=E(s_{1:n}),\quad r=C(b),\quad \hat s_{1:n}=D(r)
$$

若信道无误码：

$$
r=b\Rightarrow \hat s_{1:n}=s_{1:n}
$$

若信道有损（如比特翻转概率 $\varepsilon$、比特长度 $L$）：

$$
P(r=b)=(1-\varepsilon)^L
$$

因此不再保证正确还原，甚至可能触发解码异常（变长码失步）。

## 模块索引

1. `probability.py`：信息量、熵、条件熵、互信息及安全条件分布工具。
2. `chain.py`：`MemoryLessChain` 与 `MarkovChain`。
3. `source.py`：统一 `Source` 包装无记忆/马尔可夫信源。
4. `channel.py`：统一 `Channel` 包装无记忆/马尔可夫信道。
5. `codec.py`：`HuffmanCodec`、`huffman_encoding`、`huffman_encode/decode`。
6. `web/`：Flask 网页计算器。

## 备注

当前文档已与 `example.py`、`solve.py` 和网页端保持对应关系。后续如果继续扩展题目模板、导出功能或更多交互能力，建议同步更新本节说明。
