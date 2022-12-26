# Word2Vec

## Overview
embedding_dimensions= number_of_categories**0.25
- 实际上是使用全连神经网络实现skip-gram中的最大似然概率，来预测词与词之间的临近关系
- 改进还包括注重词序，并对词组（phrases）进行矢量化
- Simplified variant of Noise Contrastive Estimation (NCE)
- Subsampling of frequent words => 加速
- Hierarchical Softmax => only log2(W) node number


## Methods
- tf.nn.embedding_lookup 与词向量无关，只是通过一个one-hot向量v在矩阵M中找出v里1所对应位置的行 (see [link](https://www.zhihu.com/question/52250059))
- Skip-Gram 跳字模型 (Mikolov et al. 2013)
  - Efficient estimation of word representations in vector space 
  - 假设基于某个词来生成他在文本周围（窗口内）的词
  - 计算word1出现时出现word2的条件概率 p(word2|word1)
