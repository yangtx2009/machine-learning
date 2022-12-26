# Graph Embedding 图嵌入

<u>旨在将图的节点表示成一个低维向量空间，同时保留网络的拓扑结构和节点信息</u>，以便在后续的图分析任务中可以直接使用现有的机器学习算法。

## 概念
- **网络表示学习**：图嵌入和网络表示学习均表示 Graph / Network Embedding

## 挑战
  - 邻接矩阵中绝大多数是 0，数据的**稀疏性**使得快速有效的学习方式很难被应用。
  - Direct embedding methods lack the ability of generalization 直接使用embedding缺少泛化性
  
## 模型
- Skip-Gram是给定input word来预测上下文。
- CBOW (Continuous Bag of Words) 是给定上下文，来预测input word。


## 方法
| 模型 | 目标 | 输入 | 输出 |
|-----|------|------|------|
| Word2Vec | 词 | 句子 | 词嵌入 |
| DeepWalk | 节点 | 节点序列 | 节点嵌入 |

- **Random Walk**：基于随机游走的图嵌入通过使得图上一个短距的随机游走中共现的节点具有更相似的表示的方式来优化节点的嵌入。
- **DeepWalk**：DeepWalk 算法主要包含两个部分：一个随机游走序列生成器和一个更新过程。
  - 随机游走序列生成器首先在图 中均匀地随机抽样一个随机游走的根节点，接着从节点的邻居中均匀地随机抽样一个节点直到达到设定的最大长度 。
  - 对于一个生成的以为中心左右窗口为 的随机游走序列，DeepWalk 利用 SkipGram 算法通过最大化以为中心，左右为窗口的同其他节点共现概率来优化模型
  - 无法保留图中的非对称信息
- **node2vec**
  - 通过改变随机游走序列生成的方式进一步扩展了 DeepWalk 算法
  - 通过引入两个参数p和q，将宽度优先搜索和深度优先搜索引入了随机游走序列的生成过程，控制随机游走序列的跳转概率。
  - 无法保留图中的非对称信息
- **GraRep**
  - 提出了一种基于矩阵分解的图嵌入方法 (Matrix Fractorization)
  - 对于一个图G，利用邻接矩阵S定义图的度矩阵D
<img src="https://latex.codecogs.com/svg.image?D_{ij}&space;=\begin{cases}\sum_p&space;S_{ip}&space;&&space;\text{if&space;}&space;i&space;=&space;j\\0&space;&&space;\text{if&space;}&space;i&space;\neq&space;&space;j\end{cases}&space;" title="D_{ij} =\begin{cases}\sum_p S_{ip} & \text{if } i = j\\0 & \text{if } i \neq j\end{cases} " />
  - 一阶转移概率矩阵定义如下
<img src="https://latex.codecogs.com/svg.image?A&space;=&space;D^{-1}S" title="A = D^{-1}S" />
- **HOPE**
  - 对于每个节点最终生成两个嵌入表示：一个是作为源节点的嵌入表示，另一个是作为目标节点的嵌入表示。
  - 模型通过近似高阶相似性来保留非对称传递性，其优化目标为
<img src="https://latex.codecogs.com/svg.image?\min{\left\|&space;\mathbf{S}-\textbf{U}^{\textbf{s}}&space;\cdot&space;\textbf{U}^{\textbf{t}^T}&space;\right\|_F^2}" title="\min{\left\| \mathbf{S}-\textbf{U}^{\textbf{s}} \cdot \textbf{U}^{\textbf{t}^T} \right\|_F^2}" />
 
- **metapath2vec**
  - 提出了一种基于元路径的异构网络表示学习方法
- **HIN2Vec**
  - 提出了一种利用多任务学习通过**多种关系**进行节点和元路径表示学习的方法

- **SDNE**
  - 提出了一种利用自编码器同时优化一阶和二阶相似度的图嵌入算法，学习得到的向量能够保留局部和全局的结构信息。
- **DNGR**
  - 提出了一种利用基于 Stacked Denoising Autoencoder（SDAE）提取特征的网络表示学习算法。
  - 模型首先利用 Random Surfing 得到一个概率共现（PCO）矩阵，之后利用其计算得到 PPMI 矩阵，最后利用 SDAE 进行特征提取得到节点的向量表示。

## References
- [graph embedding](https://drive.google.com/file/d/1ue7r5P48NWupzuYDj6Dkt-lKKBQmNclh/view?usp=drivesdk)
- [图嵌入 (Graph Embedding) 和图神经网络 (Graph Neural Network)](https://leovan.me/cn/2020/04/graph-embedding-and-gnn/)