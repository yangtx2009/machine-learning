---
title: Data Augmentation
author: yangtx
date: 2023-04-29 06:59:00 +0100
categories: [Machine Learning, Computer Vision]
tags: [lane-detection, segmentation, object-detection]
math: true
---

## Lane Detection

### Overview
- Segmentation-based methods
- Anchor-based methods
  
  - [End-to-end traffic line detection with line proposal unit 2019](https://ieeexplore.ieee.org/document/8624563)
  - [CurveLane-NAS 2020](https://paperswithcode.com/paper/curvelane-nas-unifying-lane-sensitive)
  - [LaneATT 2020](https://github.com/lucastabelini/LaneATT)
- Row-wise methods
  - [FastDraw: Addressing the long tail of lane detection by adapting a sequential prediction network 2019](https://arxiv.org/abs/1905.04354)
  - [Ultra fast lane detection 2020](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
  - [End-to-end lane marker detection via row-wise classification 2020](https://arxiv.org/abs/2005.08630)

- Parametric prediction methods (faster but not more accurate)
  - [PolyLaneNet 2020](https://github.com/lucastabelini/PolyLaneNet)
  - [lane shape prediction with transformers (LSTR) 2021](https://github.com/liuruijin17/LSTR)

### Challenges
- Instance-level discrimination
  - **Solution 1**: predict lane points + aggregate the points into lines.
    - Label the lane lines into classes (instance ID) of a fixed number and make a multi-class classification
    - Drawback: hard to assign different points to different lane instances. Only fixed number of lanes can be detected. 

  - **Solution 2**: anchor-based methods
    - Drawback: not flexible to predict the line shape due to the fixed shape of the anchor.

- Detection of lane lines with complex topologies (fork/dense lines)


### Dataset
- MIKKI
- TuSimple
- [CULane](https://xingangpan.github.io/projects/CULane.html)
- [CurveLane](https://github.com/SoulmateB/CurveLanes)

### History
- [Lane detection in driving assistance system 2008](https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.20255)
- Spatial relationship modeling
  - Markov Random Fields (MRF)
    - [MRFNet 2016](mrfnet.md)
  - Conditional Random Fields (CRF)
    - [DenseCRF (NIPS 2011)](https://arxiv.org/abs/1210.5644)
      - https://blog.csdn.net/qq_31347869/article/details/91344524

### Benchmark
- [Papers with Code Task](https://paperswithcode.com/task/lane-detection/latest)
  - the structures are diverse and complex
  - contains various radius of curvature
  - discontinuous dotted lines
- [awesome lane detection](https://github.com/amusi/awesome-lane-detection)

---

### [SCNN (AAAI 2018)](https://drive.google.com/file/d/1ylfhFh-bwZhGHkZpi3fcdYc7tqPxjCt3/view?usp=drivesdk)

- SCNN: Sparse CNN
  
- Algorithm：
  1. slice-by-slice convolutions within feature maps, thus enabling message passings between pixels across rows and columns in a layer 将感受野在上下左右4个方向上分割, 一层层地做卷积
  2. 把卷积得到地特征作为residual传到下一层
  3. 最后预测出一个语义分割地结果
   
本质上是种autoencoder，只是在感受野上做了人为的分割，是针对lane detection这个问题的一种特殊regularization

Reference
- [SCNN: An accelerator for compressed-sparse convolutional neural networks](https://ieeexplore.ieee.org/document/8192478)

---

### [ENet-SAD (ICCV 2019)](https://drive.google.com/file/d/1Pf5szYZ_hx6McPOM3Zkxto1Qnel6ck5t/view?usp=drivesdk)

Learning Lightweight Lane Detection CNNs by Self Attention Distillation

---

### [CondLaneNet (ICCV 2021)](https://github.com/aliyun/conditional-lane-detection)

- CondLaneNet: a Top-to-down Lane Detection Framework Based on Conditional Convolution

- Proposal
  - Conditional lane detection: resolve the lane instance-level discrimination problem
  - Row-wise formulation: no limitation on line shape assumption
  - Recurrent Instance Module(RIM): deal with the detection of lane lines with complex topologies such as the dense lines and fork lines

- Methods
  - Aim: predict a collection of lanes $L={l_1,l_2,...,l_N}$. N = total number of lanes
  - Each $l_k$ is an ordered set of coordinates ($N_k$ points). $N_k$ = max number of sample points of the kth lane

  - Shape Predition
    - **Row-wise Location**: divide the input image into grids of shape Y x X. For each row, we predict the probability (sigmoid) that the lane line appears in each grid.
    - **Vertical Range**: determined by row-wisely predicting whether the lane line passes through the current row (linear layer -> binary classification)
    - **Offset Map**: predict the offset in the horizontal direction near the row-wise location for each row
    - **Shape Description**
