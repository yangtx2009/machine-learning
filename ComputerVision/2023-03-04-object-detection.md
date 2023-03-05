---
title: Object Detection
date: 2023-03-04 20:13:00 +0100
categories: [Machine Learning, Computer Vision, Detection]
tags: []     # TAG names should always be lowercase
math: true
---

## Survey
- [Towards Performing Image Classification and Object Detection with Convolutional Neural Networks in Autonomous Driving Systems: A Survey (December 2021)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9696317)

## [Fast R-CNN (ICCV 2015)](https://drive.google.com/file/d/18nDIQ_7Qk7PfVhkwqmZCNsl7Bw1P_umz/view?usp=sharing)

![](/assets/img/papers/fast-rcnn.PNG)

- R-CNN: Region-based Convolutional Network method
- Use SPPnets (Spatial pyramid pooling networks) to speed up R-CNN
  - one forward pass for all proposals (R-CNN needs to run conv-op once for each proposal)

- RoI pooling layer
  - uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of H × W
  - small feature maps are flattened to RoI feature vector
  - RoI feature vector is sent to bbox and cls FC layer branches

- Smooth L1 loss on bbox regression

$$\text{smooth}_{L_1}(x) = \left\{\begin{matrix} 0.5x^{2} & \text{if } |x|<1 \\|x|-0.5 &\text{otherwise}\end{matrix}\right.$$

---

## [Faster R-CNN (NIPS 2015): Towards Real-Time Object Detection with Region Proposal Networks](https://drive.google.com/file/d/1dXHOciAgu9CdqyZjQLOwJowRIqYWA2_h/view?usp=sharing)

- convolutional feature maps used by region-based detectors, can also be used for generating region proposals.

![](/assets/img/papers/images/faster-rcnn.PNG)

- Region Proposal Network (RPN) 
  - takes an image (of any size) as input
  - outputs a set of rectangular object proposals, each with an objectness score.
  - This feature is fed into two sibling fully-connected layers
    - a box-regression layer (reg)
    - a box-classification layer (cls).

---

## [You Only Look Once (YOLOv1, CVPR 2016): Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

---

## [RetinaNet (ICCV 2017): Focal Loss for Dense Object Detection](https://drive.google.com/file/d/19kEO8wkKiBUzYEEHVTwDVBS3tcNGyNa2/view?usp=sharing)
- **Focal Loss** (weighted cross-entropy loss)
  - down-weight easy examples and thus focus training on hard negatives.
    $$ FL(p_t)=-\alpha(1-p_t)^{\gamma}\log(p_t) $$

- **Feature Pyramid Network Backbone**
    ![](/assets/img/papers/images/retinanet.png)
    - augments a standard convolutional network with a top-down pathway and lateral connections

---

## YOLO9000 (YOLOv2, CVPR 2017): Better, Faster, Stronger

---

## [YOLOv3 (arxiv 2018): An Incremental Improvement](https://arxiv.org/abs/1804.02767)

---

## [YOLOv4 (arxiv 2020): Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)