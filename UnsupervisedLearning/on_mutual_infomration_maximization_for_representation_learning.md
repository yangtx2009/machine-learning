# [On Mutual Information Maximization for Representation Learning](https://paperswithcode.com/paper/on-mutual-information-maximization-for)

## Overview
研究了最大化共享信息MI在优化表征学习中的作用

## Conclusion
- 最大化MI并不见得能提升表征学习
- 最大化MI可以用NCE，NWJ estimator来实现，他们是MI的lower bound
- 之前的论文之所以能用estimators最大化MI，以此能优化表征学习，原因是estimator对某些dataset存在bias，所以并不能说最大化MI优化了表征学习
- 较松的bound和简单的critic有利于表征学习
- encoder架构对特征学习更重要
- Deep Metric Learning中negative sample并不是越多越好