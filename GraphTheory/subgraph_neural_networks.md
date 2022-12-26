# [Subgraph Neural Networks (NIPS 2020)](https://drive.google.com/file/d/1JLxXYxboxKHQegfeSDrLiJf-fLU5t2ZJ/view?usp=drivesdk)

## Overview
- 目的是学习分解的子图表征 learn disentangled subgraph representations
- 子图提取和网络社区检测 Subgraph extraction and network community detection
- 子图间传递神经信息 propagates neural messages between the subgraph's components
- 从底层图中随机抽取的锚点补丁 randomly sampled anchor patches from the underlying graph


## Definitions
- 全图 <img src="https://latex.codecogs.com/svg.image?G=(V,E)">
- 子图 <img src="https://latex.codecogs.com/svg.image?S=(V',E')">，并拥有一个标注 <img src="https://latex.codecogs.com/svg.image?y_S">并可能含有相连的多个子图部件（components）<img src="https://latex.codecogs.com/svg.image?S^{(C)}">
- 消息传播 MSG <img src="https://latex.codecogs.com/svg.image?m_{ij}^l = MSG(h_i^{l-1},h_j^{l-1})">
- AGG: 邻居 <img src="https://latex.codecogs.com/svg.image?\mathcal{N}_{v_{i}}"> 信息聚合函数  
- UPDATE: 将聚合的信息结合上一层的表征 <img src="https://latex.codecogs.com/svg.image?h_i^{l-1}"> 并生成l层的表征 <img src="https://latex.codecogs.com/svg.image?h_i^l">

## Method
- 子图间的消息传递架构 a neural message passing architecture <img src="https://latex.codecogs.com/svg.image?E_S"> 。它会生成一个 <img src="https://latex.codecogs.com/svg.image?d_S"> 维的子图表达 <img src="https://latex.codecogs.com/svg.image?z_S \in \mathbb{R}^{d_s}">
- SubGNN使用学到的子图表达 <img src="https://latex.codecogs.com/svg.image?z_S"> 预测标注 <img src="https://latex.codecogs.com/svg.image?f(S) = \hat{y}_{S}">
- 子图的6项重要属性
![](images/SubGNN_properties.png)
![](images/SubGNN_structures.png)

- **anchor patch**
  - 从全图中随机采样的子图集合 <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_x = \{\mathcal{A}_x^{(1)}, ..., \mathcal{A}_x^{(n_A)}\}">，分为 position <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_P">，neighborhood <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_N">，structure <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_S">
  - 采样方程为 <img src="https://latex.codecogs.com/svg.image?\phi_X: (G,S^{(c)}) \rightarrow A_X">。
    - internal position sampling <img src="https://latex.codecogs.com/svg.image?\phi_{P_I}"> 随机在子图内部采样一个节点作为该子图的锚补丁<img src="https://latex.codecogs.com/svg.image?A_{P_I}">。如果子图含有多个component，则使用整合的锚补丁集合代表整个子图 <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_{P_I}">
    - border position sampling <img src="https://latex.codecogs.com/svg.image?\phi_{P_B}"> 随机采样多个节点 <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_{P_B}">。**他们被所有子图共享**。
    - internal neighborhood sampler <img src="https://latex.codecogs.com/svg.image?\phi_{N_I}"> 从一个子图部件 <img src="https://latex.codecogs.com/svg.image?S^{(c)}"> 内部随机采样多个节点 <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_{N_I}">。
    - border neighborhood sampler <img src="https://latex.codecogs.com/svg.image?\phi_{N_B}"> 从子图部件 <img src="https://latex.codecogs.com/svg.image?S^{(c)}"> 的k-hop邻居中 随机采样多个节点 <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_{N_B}">。
    - structure sampler <img src="https://latex.codecogs.com/svg.image?\phi_S"> 被全网internal和border共享，获得两个锚补丁集合 <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_{S_I}"> 和 <img src="https://latex.codecogs.com/svg.image?\mathcal{A}_{S_B}">。
  - 锚补丁编码 encoding
    - <img src="https://latex.codecogs.com/svg.image?\psi_N">和<img src="https://latex.codecogs.com/svg.image?\psi_P"> 直接映射节点嵌入
    - <img src="https://latex.codecogs.com/svg.image?\psi_S">返回的是一个结构的表征，方式是用w-定长 triangular random walks 生成一条路径。再用bi-LSTM学习一个隐藏状态作为该结构的表征。

![](images/SubGNN_anchor_patch.png)

- **子图表征学习**
  - 将神经消息从 锚补丁（anchor patch）传播到 子图部件（subgraph component）
  - 再聚合合成的表征到最后的子图嵌入 aggregating the resultant representations into a final subgraph embedding <img src="https://latex.codecogs.com/svg.image?MSG_X^{A\rightarrow S} = \gamma_X(S^{(c)}, A_X)\cdot a_X">
    - X 表示channel，为P, N, S之一
    - <img src="https://latex.codecogs.com/svg.image?\gamma_X: (S^{(c)}, A_X) \rightarrow [0,1]"> 为相似性方程 similarity function，用来计算 anchor patch 和 subgraph component的相似性。
      - P: shortest path (SP)
      - N: conditional constant
      - S: normalized dynamic time warping (DTW)
    - <img src="https://latex.codecogs.com/svg.image?a_X"> 为学习到的anchor patch表征
  - 获得聚合信息 <img src="https://latex.codecogs.com/svg.image?g_{X,c} = AGG_M(\{MSG_X^{A_X\rightarrow S^{(c)}} \forall A_X \in \mathcal{A}_X\})">

  - 结合聚合信息和本地隐藏表征，得到更新的表征 <img src="https://latex.codecogs.com/svg.image?h_{X,c} \leftarrow \sigma(W_X \cdot [g_{X,c}; h_{X, c}]\})">
  - **注意！** 所有这些都是对P, N, S三个channel分开做的。最后合起来 <img src="https://latex.codecogs.com/svg.image?\{P_I, P_B\}, \{N_I, N_B\}, \{S_I, S_B\}">。用<img src="https://latex.codecogs.com/svg.image?AGG_C"> 聚合3个channel。用<img src="https://latex.codecogs.com/svg.image?AGG_L"> 聚合针对某个子图的所有layer的信息。


## References
- [ArXiv](https://arxiv.org/abs/2006.10538)