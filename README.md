[中文文档](./docs/zh/README-zh.md) | [English Docs](./docs/en/README-en.md)

# pyShannon

`pyShannon` 是一个面向离散信息论建模与仿真的本地 Python 工具库。

当前项目主要覆盖：

1. 离散概率分布上的信息量、熵、条件熵、互信息计算
2. 无记忆链与高阶马尔可夫链
3. 无记忆信源与马尔可夫信源
4. 无记忆信道与马尔可夫信道
5. 哈夫曼编码、严格解码与容错解码
6. 示例脚本、题目求解脚本、实验重写脚本
7. 本地 Flask 信息论计算器

## 环境

```text
Python >= 3.11
NumPy >= 1.25
Flask >= 3.1
```

当前本地开发环境建议直接使用 conda 环境 `pythontest`。

## 快速开始

运行完整示例：

```bash
python example.py
```

运行本地网页计算器：

```bash
python web/app.py
```

## 实验脚本

`exp/` 目录下已经按当前库接口重写了 4 份实验脚本：

1. `experiment1_stationary_sources.py`
2. `experiment2_cascade_channels.py`
3. `experiment3_huffman_image_compression.py`
4. `experiment4_reliability_vs_efficiency.py`

这些脚本的目标是替换实验报告里原先那些独立实现、接口不一致的旧代码。

## 文档

详细文档见：

1. 中文：[docs/zh/README-zh.md](./docs/zh/README-zh.md)
2. 英文：[docs/en/README-en.md](./docs/en/README-en.md)

`docs/` 中包含：

1. 理论公式说明
2. Python 侧调用方式
3. 网页端使用说明
4. 当前模块与目录结构说明

## 目录索引

1. `probability.py`：信息量、熵、条件熵、互信息、安全条件分布工具
2. `chain.py`：`MemoryLessChain`、`MarkovChain`
3. `source.py`：统一 `Source` 包装类
4. `channel.py`：统一 `Channel` 包装类
5. `codec.py`：哈夫曼编码、严格解码、容错解码
6. `example.py`：端到端功能示例
7. `solve.py`：题目求解脚本
8. `web/`：Flask 应用与前端资源
9. `exp/`：实验报告与重写后的实验脚本
