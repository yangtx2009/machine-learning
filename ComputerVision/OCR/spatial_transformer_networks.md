# [Spatial Transformer Networks (NIPS 2015)](https://drive.google.com/file/d/1fKuoKQo3drQ7elaf8Y3LP7nKCDv9FTUj/view?usp=drivesdk)

<img height=280 src="images/spatial_transformer_networks_example.png">

## Overview
- 可以处理 random translation, scale, rotation, clutter （杂乱）
- STN对于多通道图像的每个通道都应用相同的扭曲（wraping）操作

## Method
- Localisation Network
  - 生成 空间转换参数，也就是转换矩阵里的 <img src="https://latex.codecogs.com/svg.image?\theta" /> <img src="https://latex.codecogs.com/svg.image?\binom{x_i^s}{y_i^s}&space;=&space;\mathcal{T}_\theta&space;(G_i)&space;=&space;A_\theta&space;\begin{pmatrix}x_i^t&space;\\y_i^t&space;\\1\end{pmatrix}&space;=&space;\begin{bmatrix}\theta_{11}&space;&&space;\theta_{12}&space;&&space;\theta_{13}&space;\\\theta_{21}&space;&&space;\theta_{22}&space;&&space;\theta_{23}&space;\\\end{bmatrix}&space;\begin{pmatrix}x_i^t&space;\\y_i^t&space;\\1\end{pmatrix}" />
- Grid Generator
  - 根据空间转换参数，生成一个sampling grid（采样网格）
- Sampler
  - 根据采样网格，进行空间（仿射）变换，并输出结果


## References
- [NIPS](https://proceedings.neurips.cc/paper/2015/file/33ceb07bf4eeb3da587e268d663aba1a-Paper.pdf)
- [ArXiv](https://arxiv.org/abs/1506.02025)
- [Review: STN — Spatial Transformer Network (Image Classification) (towards data science)](https://towardsdatascience.com/review-stn-spatial-transformer-network-image-classification-d3cbd98a70aa)
- [Spatial Transformer Networks (知乎)](https://zhuanlan.zhihu.com/p/37110107)