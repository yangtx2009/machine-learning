# Recommendation
personalization (for users' taste)


## Categories
- Content-based 基于内容的推荐
- Collaborative Filtering 基于协同过滤的推荐算法
  - Memory-based CF
    - User-based CF
    - Item-based CF
  - Model-based CF
- Association Rule-based 基于关联规则的推荐
- Utility-based 基于效用的推荐
- Knowledge-based 基于知识的推荐

## Metrics
- live A/B testing

## Dataset
- [Movie Lens Dataset](http://grouplens.org/datasets/movielens/)
- [KuaiRec 快手视频分享日志数据](https://kuairec.com/)
- [Tenrec 腾讯视频推荐与图文推荐数据](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html)
- [阿里巴巴用户行为数据集](https://tianchi.aliyun.com/dataset/dataDetail?dataId=81505)
- [Jester 笑话评分系统](https://eigentaste.berkeley.edu/dataset/)
- [Book-Crossings 图书评分数据集](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
- [netflix 电影评分数据集](https://www.kaggle.com/netflix-inc/netflix-prize-data)
- [Last.FM 音乐评价数据集, 包含社交关联](https://grouplens.org/datasets/hetrec-2011/)
- [Retailrocket 电商数据集](https://www.kaggle.com/retailrocket/ecommerce-dataset)
- [FourSquare 地理社交场馆评价信息](https://archive.org/details/201309_foursquare_dataset_umn)

## 1. Content-based 基于内容的推荐

## 2. Collaborative Filtering 基于协同过滤的推荐算法
-  利用用户的历史喜好信息计算用户之间的距离
- 为一用户找到他真正感兴趣的内容的好方法是首先找到与此用户有相似兴趣的其他用户，然后将他们感兴趣的内容推荐给此用户
### 2.1. Memory-based CF
#### 2.1.1. User-based CF
#### [2.1.2. Item-based CF](https://zhuanlan.zhihu.com/p/31807038)
  - 描述a,b物品的相似性 
  <p align="center"><img align="center" src="https://latex.codecogs.com/svg.image?W_{ab}&space;=&space;\frac{\left|&space;N(a)\bigcap&space;N(b)\right|}{\sqrt{\left|&space;N(a)\right|\left|&space;N(b)\right|}}" title="W_{ab} = \frac{\left| N(a)\bigcap N(b)\right|}{\sqrt{\left| N(a)\right|\left| N(b)\right|}}" /></p>

  - 用户u买物品b的概率
  <p align="center"><img align="center" src="https://latex.codecogs.com/svg.image?P_{ub}&space;=&space;\sum_{a\in&space;N(u)\bigcap&space;S(b,K)}W_{ab}R_{ua}" title="P_{ub} = \sum_{a\in N(u)\bigcap S(b,K)}W_{ab}R_{ua}" /></p>

### 2.2. Model-based CF
- Aspect Model
- pLSA
- LDA
- Clustering
- SVD
- Matrix Factorization

## 3. Association Rule-based 基于关联规则的推荐
- 把已购商品作为规则头，规则体为推荐对象。关联规则挖掘可以发现不同商品在销售过程中的相关性
- 即买A商品后往往会买B商品

### [3.1. Apriori](https://medium.com/machine-learning-researcher/association-rule-apriori-and-eclat-algorithm-4e963fa972a4)
见Softwaretechnik II
### 3.2. FP-Growth (Frequent Pattern)
### [3.3. ECLAT (Equivalence CLAss Transformation 频繁项集挖掘算法)](https://blog.csdn.net/my_learning_road/article/details/79728389)


## 4. Utility-based 基于效用的推荐
基于效用推荐的好处是它能把非产品的属性，如提供商的可靠性( Vendor Reliability)和产品的可得性( Product Availability)等考虑到效用计算中

## 5. Knowledge-based 基于知识的推荐
效用知识( Functional Knowledge)是一种关于一个项目如何满足某一特定用户的知识，因此能解释需要和推荐的关系，所以用户资料可以是任何能支持推理的知识结构，它可以是用户已经规范化的查询，也可以是一个更详细的用户需要的表示