# [Object Detection in 3D Point Clouds via Local Correlation-Aware Point Embedding](https://drive.google.com/file/d/1OFAG80c_-dDkxRbEEmqieeeZI5uzXpCs/view?usp=drivesdk)

## Idea
- 在 [Frustum PointNets 2018](contents/ML/PointCloud/frustum_pointnets.md) 的基础上进行优化，对相邻点求嵌入表达（embedding），模拟3D卷积操作
- 使得每个点不仅包含他自己的信息，还包含邻近点信息

## Proposal
- 使用KNN找出每个点的K个相邻点 <img src="https://latex.codecogs.com/svg.image?N_i=\left\{p_{j1},p_{j2},\dots , p_{jK}\right\}" />
- 计算 local patch <img src="https://latex.codecogs.com/svg.image?\mathcal{D}(p_i,N_i)" />
- 使用全连网络和max pooling以输出点云特征（维度：64） 
    <img src="https://latex.codecogs.com/svg.image?\mathcal{F}(p_i,N_i)=\underset{p_j\in N_i}{\text{MP}}f(\mathcal{D}(p_i,p_j))" />

![](images/local_point_embedding_result.png)

## References
- [KITopen: Institut für Anthropomatik und Robotik (IAR)](https://publikationen.bibliothek.kit.edu/1000126362)