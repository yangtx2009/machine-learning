# [What Can Neural Networks Reason About? (ICLR 2020)](https://drive.google.com/file/d/1ApY68JtZrzBV8t8i13BWL3q5ZBLGX-qK/view?usp=drivesdk)

## Overview
- 描述神经网络对何种推理任务表现得好 characterize which reasoning tasks a network can learn well
- 基于一种观察：推理过程类似于算法。如果一个推理算法很好匹配网络地计算图，则网络只需要学习简单算法步骤就可以模拟出reasoning过程
- 研究计算结构多好地匹配算法结构 study how well its computation structure aligns with the algorithmic structure of the relevant reasoning process

## Backgrounds
- 不同reasoning task类型
  - intuitive physics
    - predicting the time evolution of physical objects
    - mathematical reasoning
    - visual IQ tests
  - visual question answering
  - shortest paths (dynamic programming DP)
- GNN可以通过学习一个很简单的步骤，即该算法中的松弛过程，来很好地模拟Bellman-Ford算法，而与此相对应地，MLP却必须去模拟整个for循环才可以模拟Bellman-Ford算法。
  

## Tasks
- 定义算法匹配性 define algorithmic alignment
- 推导采样复杂度界限 derive a sample complexity bound that decreases with better alignment（高算法匹配性可以降低采样复杂度界限）

## References
- [OpenReview ICLR 2020](https://openreview.net/forum?id=rJxbJeHFPS)