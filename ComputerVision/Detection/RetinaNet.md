# [RetinaNet: Focal Loss for Dense Object Detection 2017](https://drive.google.com/file/d/19kEO8wkKiBUzYEEHVTwDVBS3tcNGyNa2/view?usp=sharing)
- **Focal Loss** (weighted cross-entropy loss)
  - down-weight easy examples and thus focus training on hard negatives.
    <img src="https://latex.codecogs.com/svg.image?FL(p_t)=-\alpha(1-p_t)^{\gamma}\log(p_t)" title="" />
- **Feature Pyramid Network Backbone**
    ![](images/retinanet.png)
    - augments a standard convolutional network with a top-down pathway and lateral connections
   