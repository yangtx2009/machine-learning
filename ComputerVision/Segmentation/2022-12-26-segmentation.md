---
title: Segmentation
date: 2023-03-04 20:13:00 +0100
categories: [Machine Learning, Computer Vision]
tags: []     # TAG names should always be lowercase
---

## Mask R-CNN (2017)
Fast R-CNN + a branch for predicting an object mask in parallel with the existing branch for bounding box recognition
- Mask-Head, 以像素到像素的方式来预测分割掩膜, 并且效果很好
- ROI Align替代了ROI Pooling，去除了RoI Pooling的粗量化
- 分类框与预测掩膜共享评价函数, 对分割结果有所干扰

### Methods
- Binary ROI mask
- RoIAlign

### Reference
- [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [Paper with Note](https://drive.google.com/file/d/19Ibg51XMm0yI0-6URW-kL46UitIhmqVC/view?usp=drivesdk)