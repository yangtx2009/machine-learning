<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">

# [FOTS: Fast Oriented Text Spotting with a Unified Network (CVPR 2018)](https://drive.google.com/file/d/1ZZio-dRNO_QnqyDP1T15AWBrGNwCkjI4/view?usp=drivesdk)


## Overview
- 目的是解决两个问题
  - 两阶算法使得出现多个字符区域时，计算过于缓慢
  - 忽略了detection和recognition间视觉线索的相关联系 ignores the correlation in visual cues shared in detection and recognition
- 同时执行检测和识别，以加快速度
- 提出关于了**RoIRotate**算法来将卷积特征分享给 检测 和 识别 两个模块
- 第一个实现旋转文字检测与识别的端到端算法，可达 22.6 fps

## Method
- 共享特征
  - 输出1/4原图片大小的特征图
  - 直接被detection和recognition读取，recognition不需要额外学习视觉特征
- 文字检测分支
  - 输出5通道，分别是bbox的上下左右四个边界的位置，以及旋转角
  - 最后使用NMS，使用一个阈值去掉负样本，NMS的阈值也是根据统计而来的
  - 目标函数分为两个terms
    - 像素级分类Cross Entropy损失 (pixel-wise classification loss for a down-sampled score map)<img src="https://latex.codecogs.com/svg.image?L_{cls}=\frac{1}{|\Omega|}\sum_{x\in \Omega}H(p_x,p_x^*)"/>
    - 回归损失 <br><img src="https://latex.codecogs.com/svg.image?L_{reg}=\frac{1}{|\Omega|}\sum_{x\in\Omega}IoU(R_x,R_x^*)+\lambda_\theta(1-cos(\theta_x,\theta_x^*))"/>
    - 完整检测函数 <br><img src="https://latex.codecogs.com/svg.image?L_{detect}=L_{cls}+\lambda_{reg}L_{reg}"/>

 - RoIRotate
   - 目的是根据检测到的bbox，对shared feature进行变换
   - 保持高度和长宽比一致，长度随文字多少变化，使用bilinear interpolation
   - 第一步：根据bbox计算仿射变换参数
   - 第二步：对shared feature map进行仿射
 - 文字识别分支
   - 训练时，不直接使用检测结果，而是ground truth，类似于teacher forcing
   - bi-directional LSTM + CTC decoder
   - 每一列作为一个特征，输入到LSTM中
   - 目标函数
     - <img src="https://latex.codecogs.com/svg.image?L_{recog}=-\frac{1}{N}\sum_{n=1}^N \log p(y_n^*|x)"/>
![](../VideoAnalysis/images/fots.png)

## Dataset
- ICDAR 2015
- ICDAR 2017 MLT
- ICDAR 2013

## References
- [PaperswithCode](https://paperswithcode.com/paper/fots-fast-oriented-text-spotting-with-a)
- [jiangxiluning/FOTS.PyTorch (PyTorch)](https://github.com/jiangxiluning/FOTS.PyTorch)  <i class="fa fa-github"></i>
- [Pay20Y/FOTS_TF (TensorFlow)](https://github.com/Pay20Y/FOTS_TF) <i class="fa fa-github"></i>
- [ xieyufei1993/FOTS (PyTorch)](https://github.com/xieyufei1993/FOTS) <i class="fa fa-github"></i>
- [yu20103983/FOTS (TensorFlow)](https://github.com/yu20103983/FOTS) <i class="fa fa-github"></i>
- [Demo Video](https://www.youtube.com/watch?v=F7TTYlFr2QM) <i class="fa fa-youtube-play" style="color:red"></i>