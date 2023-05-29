---
title: Few-Shot Learning
author: yangtx
date: 2023-04-29 06:59:00 +0100
categories: [Machine Learning, Computer Vision, Few-Shot Learning]
tags: [object-detection, few-shot-learning]
math: true
---

## Few-Shot Learning

### Concepts
- Few-shot Learning
  - Definition: 
    - a computer program learns from **very small** experience $E$ with respect to some classes of task $T$ and performance measure $P$
    - If its performance can improve with $E$ on $T$ measured by $P$.
  - Scenario:
    - A **support set** represents the small dataset used in the training phase, which generates reference information for the second phase of testing.
    - The **query set** is the task on which the model actually needs to predict.
    - The query set classes never appear in the support set
- Meta-Learning
- Transfer Learning

### Challenges
- Inaccurate data distribution assessment $\rightarrow$  **data augmentation**
  - maximizing the exploration of data distributions with limited information
  - explore migratable intra-class or inter-class features
  - customize specific images using generators
- Feature reuse sensitivity $\rightarrow$  **transfer learning**
  - continuous accumulation of a priori knowledge by sampling largescale auxiliary datasets.
  - :exclamation: generally limited by the characteristics of current tasks and has a poor generalization to future tasks
  - the limited amount of training data, domain variations, and task modifications are the key factors that cause the model to fail to transfer well from the source domain to the target domain
- Generality of future tasks $\rightarrow$  **meta-learning**
  - learns to quickly build mappings from known tasks to target models in previously unseen tasks by **double sampling the task and data**
  - **C-way K-shot 问题**: 训练集中包含了很多的类别，每个类别中有多个样本。在训练阶段，会在训集中随机抽取 C 个类别，每个类别 K 个样本（总共 $C\times K$ 个数据），构建一个meta-task，作为模型的支撑集（support set）输入，再从这 C 个类中剩余的数据中抽取一批（batch）样本作为模型的预测对象（batch set）。这种机制使得模型学会不同 meta-task 中的共性部分，比如如何提取重要特征及比较样本相似等，忘掉 meta-task 中 task 相关部分。
  - :exclamation: meta-learning has proven effective only when the testing and training tasks are relatively similar
  - sensitive to network structure and requires fine tuning of hyperparameters
  <center><img src="/assets/img/papers/few-shot-learning_meta-learning.png"/></center>
- Defects of single-modal information $\rightarrow$ **multimodal learning**
  - getting information from other modalities

### Techniques of Few-Shot Object Detection
- Data Augmentation
  - Hand-Crafted Rules
    - Data Level: random cropping, erasure, filling, ...
    - Feature Level
  - Learning Data Processing
    - DARTS: abstract the data augmentation into multiple sub-strategies
    - adversarial feature phantom network-AFHN: The phantom diversity and discriminative features are conditional on a small number of labelled samples.
    - meta-learner: generate a network to learn similarities and differences between images end-to-end by fusing pairs of images 
    - MetaGAN: form generalizable decision boundaries between different classes
    - ...
- Transfer Learning
  - Pre-training and Fine-Tuning
  - Cross-Domain Few-shot Learning
- Meta-Learning
  - Learning model parameters
    - use meta-learning to train a hyperparameter generator
    - MAML, Reptile, FOMAML, Meta-SGD, TAML
    - MetaNAS
    - 旨在通过模型结构的设计快速在少量样本上更新参数，直接建立输入 x 和预测值 P 的映射函数
  - Learning metric algorithm
    - **siamese neural network 孪生网络**: the input to the model composes of a set of positive or negative sample pairs, and the model needs to evaluate the similarity of the images during inference stage. 一个双路的神经网络，训练时，通过组合的方式构造不同的成对样本，输入网络进行训练，在最上层通过样本对的距离判断他们是否属于同一个类，并产生对应的概率分布。在预测阶段，孪生网络处理测试样本和支撑集之间每一个样本对，最终预测结果为支撑集上概率最高的类别。
    - **triple loss**: deal with more than pairs input. It requires positive samples, negative samples, and anchor samples to be available at the same time
    - **prototype network 原型网络**: find the most representative sample as a prototype. 每个类别都存在一个原型表达，该类的原型是 support set 在 embedding 空间中的均值。
    - **matching networks 匹配网络**: maps few-shot datasets and unlabeled data to vectors in the embedding space. 为支撑集和 Batch 集构建不同的编码器，最终分类器的输出是支撑集样本和 query 之间预测值的加权求和
    - **relational network 关系网络**: similarity is calculated by using a neural network 训练一个网络来学习距离的度量方式，在 loss 方面也有所改变，考虑到 relation network 更多的关注relation score，更像一种回归，而非 0/1 分类，所以使用了MSE取代了cross-entropy
  - Learning To Transmit Information
- Multimodal complementary learning
  - Multimodal embedding
<center><img src="/assets/img/papers/few-shot-multimodal.png"/></center>
  - Generate semantic information from images

<center><img src="/assets/img/papers/summary-of-few-shot-object-detection.png"/></center>


### References
- A Comprehensive Survey of Few-shot Learning: Evolution, Applications, Challenges, and Opportunities
  - [arxiv](https://arxiv.org/abs/2205.06743)
  - [doc](https://drive.google.com/file/d/1i4q9JkpjYUhcLZ6SXleeIvd5QPI2md51/view?usp=drivesdk)
- [小样本学习（Few-shot Learning）综述](https://zhuanlan.zhihu.com/p/61215293)