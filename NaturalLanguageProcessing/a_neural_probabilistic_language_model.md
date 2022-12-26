# [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

两个全连网络,第一个将one-hot单词压缩成m维特征向量,第二个网络基于前文单词向量输出之后一个词的概率,最小化条件联合几率,以此学习出m维特征映射

- Pros:
  - learn the joint probability function of sequences of words
  - fight the curse of dimensionality with distributed representations
- Cons:
  - still cannot obtain satisfying representations of phrases and sentence