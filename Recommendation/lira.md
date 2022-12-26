# [LIRA](https://drive.google.com/file/d/1yjFZhQx-00mt2GBmgVY2daTqSpouizLv/view?usp=drivesdk)

## Challenges 挑战
1. 如何发掘用户间的重叠的兴趣
2. 如何决定推荐集的组合
3. Exploration/Exploitation Tradeoff

## Solutions 解决方法
- 用向量空间表达文件, dictionary (p number)+corpus (p-dim weighted space)
- 生成词空间
  - 忽略非常常见的单词
  - 忽略根词stem
  - TFIDF计算词权重
