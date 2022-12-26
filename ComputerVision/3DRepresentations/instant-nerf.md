# [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://drive.google.com/file/d/1NQpe2Rq4-n2lzwHFrAPl2wUQCz3umgKp/view?usp=sharing)
采用多分辨率哈希编码的即时神经图形基元

![](https://developer-blogs.nvidia.com/wp-content/uploads/2022/05/NeRF_Output.jpg)

## SOTA的缺陷
- 对于图像/3D信息表达，传统方法存储的是结构化数据，计算是干净的公式，与计算分离的

### 网络权重法
- 神经网络计算与数据混到了一起，典型如Nerf，radience field (辐射场) 数据信息存储到了网络权重里
- 但信息完全在网络权重里导致训练非常慢，效率低，网络表达能力也受训练的限制

### 树形结构法
- parametric encoding方式把latent feature用结构化方式存储，例如存到3D grid上，表达能力不受网络权重数量的限制，每次back propogate的参数只和3D grid对应的cell以及小网络相关。训练时间大幅缩短。
- 3D grid这种结构化数据，其实也非常浪费，因为三维模型只有表面信息有意义，绝大多数的cell都是空的
- 分层的树形数据结构能减少内存和训练数据量，但在训练过程中动态调整树的结构开销也不小，同样稀疏的数据结构需要动态更新，消耗很大。

## ReNF
![](https://cdn.wccftech.com/wp-content/uploads/2022/03/Instant-NeRF-Pillars-1480x833.jpg)
- 原版NeRF输入为一个5D特征，分别是
  - x,y,z (3维空间中要显示的点x)
  - θ,φ (照到点显示的点x的光线角度d) 
- 输出为
  - c (点x的颜色RGB)
  - σ (点x的体积密度 volume density，可理解为光线在该点终止的概率)
- 通过训练一个全连网络，可以学习到输入与输出间的映射关系
- 通过传统 Volume Rendering 技术，可以在2D上渲染出3D图像
- 位置x使用sinus positional encoding来增加对高频变化 （high-frequency variaion)的表达能力
- 使用hierarchical volume sampling，即使用两个网络coarse+fine，来增加渲染效率，减少对空白区域的渲染

## 新策略
- 学习一个无激活层的全连网络，作为embedding layer y = enc(x; 𝜃)，参数为𝜃
- 哈希法: LOD哈希表(最多保存T行特征)保存3D grid的特征y（特征大小为F），哈希表（TxF）保存了位置信息的同时保证了density，效率最高。通过点x的位置，根据hash方程就可以算出表格中对应特征的位置。
- 将空间网格化(gridding)，对每个网格(voxel)的顶点计算特征y。不同的网格间隔对应多解析度 (multi-resolution)，相当与pyramid。一共使用L层resolution level来获取不同的网格间距的特征，并保存在独立的hash表中。
- 空间中一个点x对应每个resolution layer里的4个点，通过linear interpolation得到一个合成的特征（大小仍为F），将所有resolution layer所得特征合并成一个特征y (大小为LF+E) 作为输入向量。E为辅助相连auxiliary input，可忽略。
- 最后如NeRF一样输入两个全连，分别预测density和color
- 使用Accelerated ray marching（加速光线追踪）来提高渲染图像效率。



## 其他使用的资源
- [colmap](https://colmap.github.io/): a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline with a graphical and command-line interface. It offers a wide range of features for reconstruction of ordered and unordered image collections.
![](https://colmap.github.io/_images/incremental-sfm.png)

## 中文分析
- [论文随记｜Instant Neural Graphics Primitives with a Multiresolution Hash Encoding Abstract](https://zhuanlan.zhihu.com/p/532357369)