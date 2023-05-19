---
title: Sound Event Detection
author: yangtx
date: 2023-05-18 21:26:00 +0100
categories: [Machine Learning, Audio Processing]
tags: [sound-event-detection]     # TAG names should always be lowercase
math: true
---

## Sound Event Detection

The goal of automatic sound event detection (SED) methods is to recognize what is happening in an audio signal and when it is happening.

In general, a single sound event detection system is used to predict activities of multiple sound classes which can be active simultaneously, leading to **multi-class multi-label classification** in each segment.

### Dataset

<center><img src="/assets/img/papers/sound-event-detection-survey1.png"/></center>

- strongly-labeled
  - [TUT Sound Events 2016 dataset](https://zenodo.org/record/45759#.ZGaAqM7P1D8)
- weakly-labeled
  - [FSDnoisy18k](https://zenodo.org/record/2529934#.ZGaBf87P1D8)
  - [AudioSet](https://research.google.com/audioset/)
  - [URBAN-SED (synthetic data)](http://urbansed.weebly.com/)

---

### GMM + HMM

---

### SVM
[A SVM-Based Audio Event Detection System](https://ieeexplore.ieee.org/document/5630626)

Extract the mel-frequency features of each frames and train a multi-class SVM model or multiple one-class SVM models.

---

### CRNN
Convolutional recurrent neural networks for polyphonic sound event detection

### Transfer Learning
use pretrained model to extract embeddings from the input and onsidered as input features for a downstream task.

- VGG-ish
- SoundNet
- $L^3$-Net


### References
- Sound Event Detection: A Tutorial
  - [doc](https://drive.google.com/file/d/1_PiC0w5tPmuW2rWazzBpGPtxjs0jycXD/view?usp=drivesdk)
- Learning sound event classifiers from web audio with noisy labels
  - [arxiv](https://arxiv.org/pdf/1901.01189.pdf)
  - [github](https://github.com/edufonseca/icassp19)
- Sound event detection (Kikyo-16)
  - [github](https://github.com/Kikyo-16/Sound_event_detection)