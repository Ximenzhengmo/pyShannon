[English](../en/README-en.md) | [中文](./README-zh.md)

# [pyShannon](https://github.com/Ximenzhengmo/pyShannon) -- 一个信息论的计算库

## 简介

这是一个信息论计算相关的库，作者边上课边写的...

### 功能（持续更新）

`example.py` 提供了一些使用示例

* 计算任意离散概率分布的各类熵，任意离散分布之间的各类互信息量.
  * I(x), I(X), H(X), H(XY), H(X|Y), I(x;y), I(X;y), I(X;Y) ... 
* 计算任意阶、任意长度马尔可夫链（或离散无记忆序列）的联合概率分布，任意级输出符号之间的联合概率分布
  * 任意阶马尔可夫链 $X_1X_2X_3...X_n$ -> $P(X_k),P(X_1X_2...X_k),P(X_aX_bX_c...)$
* 模拟离散无记忆信源和马尔可夫信源的输出序列
  * 生成任意符号组合的输出序列，支持任意先验和多样化的输出格式

### 环境（持续更新）
```python
    python 3.11  numpy 1.25
```