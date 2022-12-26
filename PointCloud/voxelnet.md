# [VoxelNet 2018: End-to-end learning for point cloud-based 3D object detection](https://drive.google.com/file/d/1nuP6PCAb5isLO6P-IDceH5rFt3FbD0Tu/view?usp=drivesdk)


## Proposal
- 规定一个固定的3D范围：D, H, W
- 分成大小相同的voxel，voxel大小为固定的 <img src="https://latex.codecogs.com/svg.image?v_D" />, <img src="https://latex.codecogs.com/svg.image?v_H" />, <img src="https://latex.codecogs.com/svg.image?v_W" />
- 将每个voxel中所有点收集起来，成为一个长度为T的点集。如果voxel中含的点超过T，则只随机选取（random sampling）其中T个。
  - <img src="https://latex.codecogs.com/svg.image?V=\left\{p_i=[x_i,y_i,z_i,r_i]^T\in&space;\mathcal{R}^4\right\}_{i=1\dots&space;t}" />
  - r是反射率

- 对voxel特征加密（Voxel Feature Encoding）
  - 计算voxel的中心点 <img src="https://latex.codecogs.com/svg.image?\left\(v_x,v_y,v_z\right\)" />
  - 进而得到强化的voxel特征 <img src="https://latex.codecogs.com/svg.image?V=\left\{p_i=[x_i,y_i,z_i,r_i,x_i-v_x,y_i-v_y,z_i-v_z]^T\in&space;\mathcal{R}^7\right\}_{i=1\dots&space;t}" />，相当于加入了局部位置信息

- 通过全连层 (FCN+BN+ReLU) 得到voxel特征 <img src="https://latex.codecogs.com/svg.image?f_i\in\mathcal{R}^m" />
- Element-wise Maxpooling 后得到 locally aggregated feature <img src="https://latex.codecogs.com/svg.image?\tilde{f}\in\mathcal{R}^m" />
- 将 laf <img src="https://latex.codecogs.com/svg.image?f_i" /> 与 <img src="https://latex.codecogs.com/svg.image?\tilde{f}" /> 合并为 <img src="https://latex.codecogs.com/svg.image?f_i^{out}=[f_i^T,\tilde{f}^T]^T\in\mathcal{R}^{2m}" />（类似于Residual Connection）
- 为了避免处理sparse voxel，使用Sparse Tensor Representation输出一个只含非空的voxel的列表

<center><img src="images/voxelnet_1.png" width=500></center>

- 对voxel特征使用3D卷积 <img src="https://latex.codecogs.com/svg.image?\text{ConvMD}(c_{in},c_{out},k,s,p)" />
<center><img src="images/voxelnet_framework.png" width=500></center>

- 最后使用Region Proposal Network从3D卷积层输出的特征中预测3D BBox或分类。输出是 <img src="https://latex.codecogs.com/svg.image?(x_c^g,y_c^g,z_c^g,l^g,w^g,h^g,\theta^g)" />

## Efficient Implementation
- 初始化一个大小为 <img src="https://latex.codecogs.com/svg.image?K\times T\times 7" />的tensor作为输入特征缓存（input feature buffer）
  - K: 最大非空voxel数
  - T: voxel中最大点数
  - 7: 原始输入特征的维度
- 随机对voxel中点进行处理，使用基于hash key检查是否随机选取的点已被处理


## References
- [LiDAR point-cloud based 3D object detection implementation with colab {Part-1 of 2} 2020](https://towardsdatascience.com/lidar-point-cloud-based-3d-object-detection-implementation-with-colab-part-1-of-2-e3999ea8fdd4)
- [arxiv](https://arxiv.org/pdf/1711.06396.pdf)