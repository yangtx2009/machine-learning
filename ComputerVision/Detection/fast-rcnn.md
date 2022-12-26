# [Fast R-CNN](https://drive.google.com/file/d/18nDIQ_7Qk7PfVhkwqmZCNsl7Bw1P_umz/view?usp=sharing)

![](images/fast-rcnn.PNG)

- R-CNN: Region-based Convolutional Network method
- Use SPPnets (Spatial pyramid pooling networks) to speed up R-CNN
    - one forward pass for all proposals (R-CNN needs to run conv-op once for each proposal)
- RoI pooling layer
    - uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of H × W
    - small feature maps are flattened to RoI feature vector
    - RoI feature vector is sent to bbox and cls FC layer branches
- Smooth L1 loss on bbox regression
    <img src="https://latex.codecogs.com/svg.image?\text{smooth}_{L_1}(x)&space;=&space;\left\{\begin{matrix}&space;0.5x^{2}&space;&&space;\text{if&space;}&space;|x|<1&space;\\|x|-0.5&space;&\text{otherwise}\end{matrix}\right." title="" />



