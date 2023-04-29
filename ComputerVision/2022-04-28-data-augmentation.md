---
title: Data Augmentation
author: yangtx
date: 2023-04-28 19:34:00 +0100
categories: [Machine Learning, Computer Vision]
tags: [data-augmentation]     # TAG names should always be lowercase
math: true
---

## [Sill-Net (2022)](https://drive.google.com/file/d/1WgoH2FgnhtATofgr8mGZNkcfzsmjjecF/view?usp=drivesdk)

- Separating-Illumination Network (Sill-Net)
- remove the semantic part of images and store the illumination to the repository for augmentations.
- Separation criterien
  - semantic features can predict labels while illumination features can not $\rightarrow$ **separation module**
  - semantic features are informative to reconstruct the corresponding
template images $\rightarrow$ **mathcing and reconstruction module**
  - Illumination features contain minimal semantic information. $\rightarrow$ **augmentation module**

<center><img src="/assets/img/papers/sill-net.png" width="400"/></center>

<center><img src="/assets/img/papers/sill-net2.png"/></center>

The framework consists of three modules which are **separately trained**:
- **separation module**
  - Extractor learns the separated feature $z\rightarrow[z_{sem},z_{illu}]$
  - choose training datasets with various illumination conditions and few confounding objects 选择环境光多样，少物体重叠的数据做训练
  - **feature exchange mechanism**:
    - extract $z_{sem}$ and $z_{illu}$ from two random images $i$ and $j$
    - exchange the features of the two images 
    - use a classifier to predict the $z_{sem}$-related classes
    - 通过对**交换并叠加的特征**做分类来训练 extractor 网络
  $$
  z = rz_{sem(i)}+(1-r)z_{illu(j)}
  $$
- **matching and reconstruction module**
  - spatial transformer $\mathcal{T}$: learn the parameters of affine transformation on the deformed objects and then rectify their semantic features to the regular positions 使用空间变换器来对特征$z_{sem}$进行变换，使其还原到template的大小和位置
  - two targets
    - train a transformer $\mathcal{T}$ to minize MSE between $z_{sem}|_{x_i}$ and $z_{sem}|_{t_i}$ in semantic feature space
    - train the reconstructor to reconstruct the template image from $z_{sem}$
- **augmentation module**
  - negative Post Interventional Disagreement (PIDA): 
  $$
  L_{illu} = - PIDA = -\sum_{c=1}^M \sum_{i=1}^{N_c} \mathcal{D}(\mathbb{E}(z_{illu}|_{x_c,y_c}), z_{illu}|_{x_ci,y_c})
  $$
  - quantifies the distances between the illumination feature of each same-labeled image $z_{illu}|_{x_ci,y_c}$ and their expectation $\mathbb{E}(z_{illu}|_{x_c,y_c})$
  - 简单地说，假设illum特征在训练集内是随机的，那么意味着特征样本要在分布中尽可能相互远离彼此


References:
- [arxiv](https://arxiv.org/abs/2102.03539)

Datasets:
- German Traffic Sign Recognition Benchmark (GTSRB)
- Tsinghua-Tencent 100K (TT100K)
- Belgian Traffic Sign Classification (BTSC)
- Chinese Traffic Sign Database (CTSD)
- BelgaLogos
- FlirckrLogos-32
- TopLogo-10