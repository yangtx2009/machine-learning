# [ITEM2VEC: Neural item embedding for collaborative filtering (MLSP 2016)](https://drive.google.com/file/d/1i5wDwq6giDdvKST-ApROjXhbpqVDK3uI/view?usp=drivesdk)


## Overview
- 主要做法是把item视为word，用户的行为序列视为一个集合，item间的共现为正样本，并按照item的频率分布进行负样本采样
- 缺点是相似度的计算还只是利用到了item共现信息
  - 忽略了user行为序列信息
  - 没有建模用户对不同item的喜欢程度高低。


## References
- [[embedding系列]-item2vec](https://zhuanlan.zhihu.com/p/139765540)
- [DNN论文分享 - Item2vec](https://zhuanlan.zhihu.com/p/24339183)