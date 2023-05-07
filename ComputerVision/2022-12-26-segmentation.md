---
title: Segmentation
author: yangtx
date: 2023-03-04 20:13:00 +0100
categories: [Machine Learning, Computer Vision]
tags: [segmentation]     # TAG names should always be lowercase
math: true
---

## Segmentation

### [Mask R-CNN (2017)](https://drive.google.com/file/d/19Ibg51XMm0yI0-6URW-kL46UitIhmqVC/view?usp=drivesdk)

Fast R-CNN + a branch for predicting an object mask in parallel with the existing branch for bounding box recognition

- Methods
  - **Mask-Head** + Binary ROI mask: predicts segmentation masks to pixel-level, and works well 以像素到像素的方式来预测分割掩膜, 并且效果很好
  - **ROI Align** replaces ROI Pooling, remove the coarse quantization of RoI Pooling 去除了RoI Pooling的粗量化
  - The classification framework shares the evaluation function with the prediction mask, which interferes with the segmentation results 分类框与预测掩膜共享评价函数, 对分割结果有所干扰

- Reference
  - [arxiv](https://arxiv.org/abs/1703.06870)

---



---

### [SAM (arXiv 2023)](https://drive.google.com/file/d/1bSfd8qH_YuGYigoPKdXyLKVYR1Eo0kWj/view?usp=drivesdk)
Segment Anything

- Purpose
  - develop a **promptable model** (可提示/互动模型) and pre-train it on a broad dataset using a task that enables powerful generalization (e.g. image segmentation)

<center>
<img src="/assets/img/papers/sam1.png">
</center>

- Framework
  - Image encoder: MAE pre-trained Vision Transformer (ViT)
  - Prompt encoder
    - Input: sparse (points, boxes, text) + dense (masks)
    - text encoder (CLIP) + positional encodings (points, boxes) + mask encoder (convolutions)
  - Mask decoder
    - Transformer decoder block
    - prompt self-attention + cross-attention in two directions
    - **Segmentation + Classification**: after running two blocks, we upsample the image embedding and an MLP maps the output token to a dynamic linear classifier
  - **Resolving ambiguity**
    - the model will average multiple valid masks (e.g. 3 masks $\rightarrow$ whole, part, and subpart) if given an ambiguous prompt

- Drawback
  - does not predict categories

- References
  - [arXiv](https://arxiv.org/abs/2304.02643)
  - [dataset](https://segment-anything.com/dataset/index.html)
  - [github](https://github.com/facebookresearch/segment-anything)

---

### SSA (github 2023)
Semantic segment anything labeling engine


- Framework
  - class proposals: a list of possible class names

<center><img src="https://github.com/fudan-zvg/Semantic-Segment-Anything/blob/main/figures/SSA_motivation.png?raw=true"/><img src="https://github.com/fudan-zvg/Semantic-Segment-Anything/blob/main/figures/SSA_model.png?raw=true"/></center>

- References
  - [github](https://github.com/fudan-zvg/Semantic-Segment-Anything)
  - [demo](https://replicate.com/cjwbw/semantic-segment-anything)