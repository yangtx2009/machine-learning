# [SCNN (Sparse CNN)](https://drive.google.com/file/d/1ylfhFh-bwZhGHkZpi3fcdYc7tqPxjCt3/view?usp=drivesdk)

步骤：
1. 将感受野在上下左右4个方向上分割，一层层地做卷积
2. 把卷积得到地特征作为residual传到下一层
3. 最后预测出一个语义分割地结果
   
本质上是种autoencoder，只是在感受野上做了人为的分割，是针对lane detection这个问题的一种特殊regularization


## Reference
- [SCNN: An accelerator for compressed-sparse convolutional neural networks](https://ieeexplore.ieee.org/document/8192478)