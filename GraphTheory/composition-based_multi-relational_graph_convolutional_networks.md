# [Composition-based Multi-Relational Graph Convolutional Networks (CompGCN ICLR 2020)](https://drive.google.com/file/d/1S3rd-aoRJ94nFaIIHneSZulW-JkIota4/view?usp=drivesdk)

## Overview
- 同时学习节点和关系的表示 jointly embeds both nodes and relations in a relational graph.

## Method
![](images/CompGCN.png)
- 多关系图表示为 <img src="https://latex.codecogs.com/svg.image?\mathcal{G}=(\mathcal{V},\mathcal{R},\mathcal{E},\mathcal{X},\mathcal{Z})" title="" />，其中<img src="https://latex.codecogs.com/svg.image?\mathcal{Z}&space;\in&space;\mathbb{R}^{\left|&space;\mathcal{R}\right|\times&space;d_0}" title="" />表示初始化的关系特征向量
- 组合运算符（composition operator）<img src="https://latex.codecogs.com/svg.image?e_o = \phi(e_s, e_r)" title="" />
  -  o: object, s: subjective, r: relation 
  - 无参运算类型
    - subtraction
    - multiplication
    - circular-correlation
  - 含参运算类型
    - Neural Tensor Networks (NTN)
    - ConvE
- 3种边类型：
  - 有向边 <img src="https://latex.codecogs.com/svg.image?R=(u,v,r)" title="" /> 
  - 反向边 <img src="https://latex.codecogs.com/svg.image?R_{inv}=(u,v,r^{-1})" title="" /> 
  - 自连边 <img src="https://latex.codecogs.com/svg.image?\top =(u,u,\top)" title="" />
- 中心节点v的聚合过程
  - <img src="https://latex.codecogs.com/svg.image?h_v= f\left(\sum_{(u,r)\in\mathcal{N}(v)}W_r h_u\right) = f\left(\sum_{(u,r)\in\mathcal{N}(v)}W_{\lambda{(r)}} \phi(x_u, z_r)\right)" title="" />
- relation-type specific parameter <img src="https://latex.codecogs.com/svg.image?W_{\lambda{(r)}}" title="" /> 是根据不同relation类型使用不同的权重 <br><img src="https://latex.codecogs.com/svg.image?W_{\text{dir}(r)}=\left\{\begin{matrix}W_O,&space;&r\in\mathcal{R}&space;\\W_I,&space;&r\in\mathcal{R_\text{inv}}&space;\\W_S,&space;&r=\top\end{matrix}\right." title="" />

## Dataset
- Link Prediction
  - [FB15k (Freebase 15K)](https://paperswithcode.com/dataset/fb15k)
  - [WN18RR (WordNet)](https://paperswithcode.com/dataset/wn18rr)
- Node Classification
  - [MUTAG (predict mutagenicity on Salmonella typhimurium 预测鼠伤寒沙门氏菌的致突变性)](https://paperswithcode.com/dataset/mutag)
  - [Amsterdam Museum artifacts relationship](https://data.europa.eu/data/datasets/oh3dbp9vsnnt2g?locale=en)
- Graph Classification
  - (Predictive Toxicology Challenge 预测性毒理学挑战)[https://relational.fit.cvut.cz/dataset/PTC]

## Metrics
- Mean Reciprocal Rank (MRR)
- Mean Rank (MR)
- Hits@N

## References
- [20ICLR 多关系图神经网络 CompGCN](https://zhuanlan.zhihu.com/p/109738386)
- [arXiv](https://arxiv.org/abs/1911.03082)
- [ICLR 2020](https://iclr.cc/virtual_2020/poster_BylA_C4tPr.html)