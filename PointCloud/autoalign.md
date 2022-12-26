# [AutoAlign 2022: Pixel-Instance Feature Aggregation for Multi-Modal 3D Object Detection](https://drive.google.com/file/d/1D-LM7nr0w2YkMVUVlXV1ZHr-Nh5bI6lP/view?usp=drivesdk)

## Idea
- 不需要摄像机投射矩阵（camera projection matrix），而是使用 注意力机制 (attention mechanism) 会自动学习映射关系（learnable alignment map）
- 使用 自监督跨模态特征互动模块 self-supervised cross-modal feature interaction module

## Proposal
- ResNet-50提取图像特征，输出大小为 <img src="https://latex.codecogs.com/svg.image?H/32\times W/32" />
- backbone最后输出 <img src="https://latex.codecogs.com/svg.image?F\in\mathcal{R}^{h\times w\times d}" />，flatten后变成2D特征向量 <img src="https://latex.codecogs.com/svg.image?F\in\mathcal{R}^{hw\times d}" />
- 从点云中voxelization，得到voxel特征 P
- **Cross-Attention Feature Alignment (CAFA) module**
    - 从图像特征F中获取 key <img src="https://latex.codecogs.com/svg.image?\mathbf{K}_i=f_i \mathbf{W}^K" /> 和 value <img src="https://latex.codecogs.com/svg.image?\mathbf{V}_i=f_i \mathbf{W}^V" />，在点云特征 P 中获取 queries <img src="https://latex.codecogs.com/svg.image?\mathbf{Q}_i=p_j \mathbf{W}^Q" />, 求点积，归一化后得到注意力矩阵

        <img src="https://latex.codecogs.com/svg.image?s_{i,j}=\frac{\exp(\beta_{i,j})}{\sum_{j=1}^{hw} \exp(\beta_{i,j})}" />, <img src="https://latex.codecogs.com/svg.image?\beta_{i,j}=\frac{\mathbf{Q}_j\mathbf{K}_i^T}{\sqrt{d_k}}" />

        <img src="https://latex.codecogs.com/svg.image?\hat{f}_i^{att} = \text{Att}(Q_i,K,V)=\sum_{j=1}^{hw}s_{i,j}V_j" />
    - 经过前馈网络得到预测 <img src="https://latex.codecogs.com/svg.image?\mathbf{F}^{att}=\text{FFN}(\hat{\mathbf{F}}_i^{att})" />
- **Self-supervised Cross-modal Feature Interaction (SCFI)**
    - 随机选取N个3D检测框 <img src="https://latex.codecogs.com/svg.image?\mathbf{B}^{3D}= (B_1^{3D},B_2^{3D},\dots,B_N^{3D})" />，并投射到2D平面上，得到 <img src="https://latex.codecogs.com/svg.image?\mathbf{B}^{2D}= (B_1^{2D},B_2^{2D},\dots,B_N^{2D})" />
    - 根据这N个Boxes，通过2DROIAlign和3DRoIPooling获取Boxes中的特征 
    <img src="https://latex.codecogs.com/svg.image?\mathbf{R}_i^{3D}=\text{3DRoIPooling}(\mathbf{P},B_i^{3D})" /><br>
    <img src="https://latex.codecogs.com/svg.image?\mathbf{R}_i^{2D}=\text{2DRoIAlign}(\mathbf{F},B_i^{2D})" />
    - 对图像和点云特征 做 相同的投影 <img src="https://latex.codecogs.com/svg.image?h" />, 点云还要额外做预测 <img src="https://latex.codecogs.com/svg.image?f" />: <img src="https://latex.codecogs.com/svg.image?p_1=f(h(R^{3D}))" /> <img src="https://latex.codecogs.com/svg.image?q_2=h(R^{2D})" />
    - 通过对称损失函数来比较特征距离 <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{SCFI}=\frac{1}{2}\mathcal{D}(p_1,q_2)+\frac{1}{2}\mathcal{D}(p_2,q_1)" />
![](images/autoalign.png)

## References
- [arxiv](https://arxiv.org/pdf/2201.06493.pdf)
