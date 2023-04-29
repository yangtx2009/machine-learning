---
title: Optical Character Recognition
author: yangtx
date: 2023-03-04 20:13:00 +0100
categories: [Machine Learning, Computer Vision]
tags: [ocr]     # TAG names should always be lowercase
math: true
---
## Optical Character Recognition

### Overview

The optical character recognition (OCR) can be divided into two levels:
- **Text Detection**:
  merely detect the text regiona, such as CTPN.
- **Text Recognition**:
  not only detect the regions but also the contents of texts.
  - Two-stage structure:
    - Yolo+CTC / DenseNet+CTC: the highest recognition speed per line can only reach around 0.02s.
  - One-stage structure

### Review & Summary
- https://github.com/hwalsuklee/awesome-deep-text-detection-recognition
- https://github.com/topics/text-detection-recognition
- https://zhuanlan.zhihu.com/p/100940108
- [Text Recognition in the Wild: A Survey (ACM Computing Surveys 2021)](https://drive.google.com/file/d/163kH7okKzlZr_nYcxeT9BYI_xWhmGb93/view?usp=drivesdk)
- [文字识别方法整理](https://zhuanlan.zhihu.com/p/65707543)
- [HCIILAB/Scene-Text-Recognition (华南理工大学)](https://github.com/HCIILAB/Scene-Text-Recognition) <i class="fa fa-github"></i>
- [HCIILAB/Scene-Text-End2end (华南理工大学)](https://github.com/HCIILAB/Scene-Text-End2end) <i class="fa fa-github"></i>

### Implementation
- https://github.com/YCG09/chinese_ocr
  - CTPN + DenseNet + CTC (Tensorflow+Keras)
- https://github.com/eragonruan/text-detection-ctpn
  - CTPN (Tensorflow)
- https://github.com/HusseinYoussef/Arabic-OCR
  - Arabic OCR
- [awesome-deep-text-detection-recognition](https://github.com/hwalsuklee/awesome-deep-text-detection-recognition)
- Servies
  - ![](/assets/img/papers/ocr-service.png)

### [Dataset](https://github.com/HCIILAB/Scene-Text-Recognition)
- Multi-lingual
  - [MLT (Multi-Lingual Text) 2017](https://rrc.cvc.uab.es/?ch=8)
    - Chinese, Japanese, Korean, English, French, Arabic, Italian, German and Indian
  - [MLT (Multi-Lingual Text) 2019](https://rrc.cvc.uab.es/?ch=15)
    - Chinese, Japanese, Korean, English, French, Arabic, Italian, German, Bangla and Hindi (Devanagari)
    - Street views 街景图片
    - 含有斜向和不同类型字体
    - [introduction](https://cbdar2019.univ-lr.fr/wp-content/uploads/2019/11/CBDAR2019_RRC-MLT-2019_CBDAR.pdf)
    
- **Chinese**
  - [caffe_ocr](https://github.com/senlinuc/caffe_ocr)
    - 大部分中文，少部分英语
    - 只能做Text Recognition
    - 图片分辨率统一为280x32
    - 只含灰度打印字体，无旋转
    - **Sentence level**，无**Character level**
  - [Chinese Text in the Wild(CTW)](https://ctwdataset.github.io/)
    - 只含有中文字
    - 街景，行车记录照片
    - **Character level**，无**Sentence level**
  - [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
    - 部分中文，部分英语
    - 含有斜向，弧形字体
    - https://paperswithcode.com/dataset/scut-ctw1500
    - **Word level** (类似于Total-Text)
    - 每个文字边界都由14个点，上下边缘各7个点
    - 保存格式为 
    ```
    xmin，ymin，xmax，ymax（外接矩形），pw1，ph1，...，p14，ph14
    ```
  - [ICPR MWI 2018 挑战赛](https://tianchi.aliyun.com/competition/entrance/231651/information?from=oldUrl)
    - 大部分中文，少部分英语
    - 含有斜向字体
    - e-commerce, social networking, and search
    - **Word level**/**Sentence level**
  - [LSVT-ICDAR2019 (Large-scale Street View Text with Partial Labeling)](https://ai.baidu.com/broad/introduction?dataset=lsvt)
    - 大部分中文，少部分英语
    - 街景照片
    - 含有斜向字体
    - **Character level** + **Sentence level**
  - [MSRA-TD500]
  - [MTWI](https://pan.baidu.com/share/init?surl=SUODaOzV7YOPkrun0xSz6A#list/path=%2F)
    - 大部分中文，少部分英语
    - 淘宝广告图片
    - 含有斜向字体
    - password: gox9
    - **Word level**
  - [Reading Chinese Text (ReCTS)](https://rrc.cvc.uab.es/?ch=12)
    - 大部分中文，少部分英语
    - 街景广告牌照片
    - 含有斜向和多种字体
    - **Word level**
  - [Reading Chinese Text in the Wild(RCTW-17)](http://mclab.eic.hust.edu.cn/icdar2017chinese/dataset.html)
  - [Total-Text](https://github.com/cs-chan/Total-Text-Dataset)
    - 大部分英语，少部分中文
    - 含有斜向，弧形字体
    - 街景，行车记录照片
    - **Word level**
    
- **English**
  - [Curved Text (CUTE80)](http://cs-chan.com/downloads_cute80_dataset.html)
    - 高分辨率照片
    - 街景+Tshirt图片
    - 含有斜向，变形，**弧形**和不同类型字体
  - [IIIT5K](https://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)
    - 只含英语
    - 含背景
    - 只能做Text Recognition
  - [ICDAR 2003](http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions)
    - 只含英语
    - 街景图片
    - 含有斜向和不同类型字体
  - [ICDAR 2013](http://dagdata.cvc.uab.es/icdar2013competition/?ch=2&com=downloads)
    - 只含英语
    - 街景图片
    - 不同类型字体，轻微斜向
  - [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4)
    - 只含英语
    - 街景图片
    - 含有斜向和不同类型字体
  - [MSCOCO-Text](https://bgshih.github.io/cocotext/#h2-explorer)
    - machine-printed vs. handwritten, legible vs. illgible, and English vs. non-English
    - 街景图片
    - 含有斜向和不同类型字体
    - **Word level**
  - [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/)
    - 只含数字
    - 街景图片
    - 不同类型字体，轻微斜向
    - **Character level**
  - [Synthetic Word Dataset](https://www.robots.ox.ac.uk/~vgg/data/text/)
    - 只含英语
    - 合成文字，不含背景
    - 只能做Text Recognition
  - [Synthetic Data for Text Localisation](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
    - 只含英语
    - 各种图片+合成文字
    - 含有斜向，**弧形**和不同类型字体
  - [Street View Text (SVT)](http://tc11.cvc.uab.es/datasets/SVT_1)
    - 只含英语
    - Google Street View街景图片
    - 含有斜向和不同类型字体
  - [Street View Text-Perspective (SVT-P)](https://pan.baidu.com/share/init?surl=rhYUn1mIo8OZQEGUZ9Nmrg)
    - Google Street View街景图片
    - 含有斜向，变形和不同类型字体
    - password: vnis
  - [USTB-SV1K]()

- **Deutsch**

- **French**
  - [Google FSNS(谷歌街景文本数据集)](https://rrc.cvc.uab.es/?ch=6)
    - 法国街景图片

- [**Arabic**]()
  - [A Review of Arabic Text Recognition Dataset](https://drive.google.com/file/d/1cpEQMoV5pI-3obkRUDdNF_C2bU0GpK6d/view?usp=drivesdk)
  - [APTI (Arabic Printed Text Image Database)](https://diuf.unifr.ch/main/diva/APTI/index.html)
    - 打印字体，方向不变换，不含背景
    - 只包含阿拉伯字母
    - **Word level**
  - [**AcTiv**](http://tc11.cvc.uab.es/datasets/AcTiV_1)
    - 新闻图片
    - 打印字体，方向不变换，含背景
    - **Sentence level**，无**Character level**
    - https://ieeexplore.ieee.org/document/7333911
  - [ARASTI (Arabic Scene Text Image)](https://ieee-dataport.org/open-access/arasti-database)
    - 字母图片，类似于SVHN
    - 打印字体，方向不变换，含背景
    - **Character level**
  - [KAFD (King Fahd Univeristy Arabic Font Database)](http://kafd.ideas2serve.net/)
    - 文档图片
    - 手写字体，方向不变换，不含背景
    - **Sentence level**
  - [HACDB (Handwritten Arabic Characters Database for Automatic Character Recognition)](http://repository.uob.edu.ly/handle/123456789/544)
    - 字母图片，类似于MNIST
    - 手写字体，方向不变换，不含背景
    - **Character level**
  - [IFN/ENIT](http://www.ifnenit.com/)
    - 手写字体，方向不变换，不含背景
    - 只能做Text Recognition
    - **Word level**
  - [SmartATID](https://sites.google.com/site/smartatid/)
    - 文档图片
    - 手写字体，方向有倾斜，模糊失焦，不含背景
    - **Sentence level**
  
- **Japanese**
  - [ICDAR 2017](https://rrc.cvc.uab.es/?ch=7&com=introduction)
    - 大部分日语，少部分英语
    - 大阪市街景照片
    - 含有斜向，弧形和不同类型字体

---

### [ASTER (TPAMI 2018)]()
An Attentional Scene Text Recognizer with Flexible Rectification

- References
  - [Github (Tensorflow 1.4)](https://github.com/bgshih/aster)
  - [Github (PyTorch)](https://github.com/ayumiymk/aster.pytorch)

---

### [ABCNet (CVPR 2020)](https://drive.google.com/file/d/1ZzdfzqJBuSGobJnIeIWl2EJSC-ytynMS/view?usp=drivesdk)

- Paper title: ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network
- Overview
  - Bezier Curve Detection and BezierAlign + CTC
  - single-shot, anchor-free
  - 相比之前的论文使用text spotting，本文自适应拟合巴塞尔曲线，仅使用几个控制点（control points）就可以描述文字的复杂排布，减少了计算成本 adaptively fit arbitrarily-shaped text by a parameterized Bezier curve (**BezierAlign layer**)
  - ABCNet还精简了pipeline，加快了计算速度
  - 高达 22.8 fps
  - 支持中英文识别
  - 之前的TextAlign和FOTS，默认文字轮廓是直边四边形，可以被认为是ABCNet的特例
  - 省去了[2D attention](towards_end-to-end_text_spotting_in_natural_scenes.md) 中的复杂变换
  - 对于任意文字排布问题，有两种解决算法：
    - segmentation-based methods：图片分割
    - regression-based methods：预定义文字排布形状，并回归分析形状模型
  - 官方的AdelaiDet库需要依赖Facebook的Detectron2，而Detectron2中有个C编译的`_C.cpython-37m-x86_64-linux-gnu.so`只能在Linux中运行，而ABCNet2中将BezierAlign编译到了一个`.cu`文件中，可能也会影响模型跨平台的迁移
    
  ![](/assets/img/papers/text_detection_recognition_e2e_abcnet.png)


- Method
  - Bezier curve
    - Bezier curve 被描述为 Bernstein Polynomials 作为的基函数的线性组合：
    $$
    c(t)=\sum_{i=0}^{n} b_i B_{i,n}(t), 0\leq t\leq 1
    $$
    - 本方法使用cubic Bezier curve，即复杂度（degree n）为3，4个控制点定义一条线。一个bbox只要8个控制点，即上下各一条巴塞尔曲线。
    - Bernstein Polynomials：
    $$
    B_{i,n}(t)=\binom{n}{i}t^i(1-t)^{n-i},i=0,...,n
    $$
  - 巴塞尔Ground Truth生成
    - CTW1500上下边各7个点，Total-Text上下边各5个点
    - 目的是将普通的基于多边形标注（polygonal annotations，m个点 <img src="https://latex.codecogs.com/svg.image?\{(p_{x_i},p_y_i)\}_{i=1}^m" /> ）转换为巴塞尔曲线标注
    - 使用least square method拟合巴塞尔曲线
    $$
    \begin{bmatrix}B_{0,3}(t_0) & \cdots & B_{3,3}(t_0) \\B_{0,3}(t_1) & \vdots  & B_{3,3}(t_1) \\\vdots  & \ddots & \vdots \\B_{0,3}(t_m) & \cdots  & B_{3,3}(t_m) \\\end{bmatrix}\begin{bmatrix}b_{x_0} & b_{y_0} \\b_{x_1} & b_{y_1} \\b_{x_2} & b_{y_2} \\b_{x_3} & b_{y_3} \\\end{bmatrix}=\begin{bmatrix}p_{x_0} & p_{y_0} \\p_{x_1} & p_{y_1} \\\vdots & \vdots \\p_{x_m} & p_{y_m} \\\end{bmatrix}
    $$
  - BezierAlign
    - 过去的non-segmentation采样方法大多使用长方形采样（RoI Pooling, RoI-Rotate, Text-Align-Sampling）
    - 本文的BezierAlign中，每列的边界与上下曲线边界正交（orthogonal，即垂直），采样点（交点）等距
    - 输出的是长方形feature map，大小 $h_{out} \times w_{out}$
    - 为了计算每个输出feature map像素在曲线上的对应点，需要算出其在输出feature map相对位置参数t 
    $$t=\frac{g_{iw}}{w_{out}}$$
    - 根据位置参数t，带入$c(t)$，算出上下延曲线上的位置 tp 和 bp
    - 进而算出原特征空间的采样点
    $$op=bp\cdot\frac{g_{ih}}{h_{out}}+tp\cdot (1-\frac{g_{ih}}{h_{out}})$$
    - 根据坐标op，使用bilinear interpolation实现插值（映射）
  - Recognition branch
    - BezierAlign输出的就是矫正好的图像
    - 识别分支使用LSTM输出文字
    - 最后的loss使用CTC Loss
  <img src="/assets/img/papers/abcnet_structure.png">

- References
  - [ArXiv](https://arxiv.org/abs/2002.10200)
  - [Github (Adelaidet, PyTorch)](https://github.com/aim-uofa/AdelaiDet)
  - [Github (Chinese)](https://github.com/Yuliang-Liu/ABCNet_Chinese)
  - [加强版ABCNet v2](./abcnet_v2.md)

---

### [ABCNet v2 (CVPR 2021)](https://drive.google.com/file/d/1ZzdfzqJBuSGobJnIeIWl2EJSC-ytynMS/view?usp=drivesdk)
- ABCNet v2: Adaptive Bezier-Curve Network for Real-time End-to-end Text Spotting
- Overview
  - 高达30~50fps
  - 相比会议版的ABCNet v1，v2加强了feature extractor，detection branch，recognition branch和end-to-end training
    - 加入了迭代双向特征 incorporates iterative bidirectional features
    - 使用了坐标编码方法 coordinate encoding approach
    - 在识别分支中增加**字符注意力模块** character attention module
    - 提出了自适应端到端训练策略 Adaptive End-to-End Training (AET) strategy

- Method
  - Structure
    - Bezier curve detection
    - the coordinate convolution module
    - BezierAlign
    - the light-weight attention recognition module
    - the adaptive end-to-end training strategy
    - text spotting quantization
  - **CoordConv**
    - 传统卷积很难学习在直角坐标系下 坐标（coordinate）间的映射，并在one-hot像素空间下定位
    - 为解决此问题，可以将 相对坐标（relative coordinates） 与 像素值 连接（concatenate），即coordinate encoding
    - 该方法直接将坐标信息引入视觉特征的学习，提高了准确率
  - BezierAlign 依然沿用v1的方法，计算相对位置参数t，并转换到巴塞尔曲线方程中
  - Attention-based Recognition Branch
    - 相比v1中的LSTM直接输出字符预测，v2**取消了CTC loss**，LSTM只提供特征 $h_s$, 并作为输入值，输入到**注意力识别模块**
    - 注意力机制每次读取
      - 裁剪过的LSTM状态特征向量 $h_s$
      - 上次的字符预测 $y_{t-1}$（c-category softmax，中英文下，c=5462。英文下c=96）
      - 前一个隐藏状态 $h_{t-1}$
    - 注意力weight根据以下公式计算
      - 注意力权重特征：$e_{t,s}=\mathbf{K}^T\tanh(\mathbf{W}h_{t-1}+\mathbf{U}h_s\mathbf{b})$
      - 归一化的注意力权重：$a_{t,s}=\frac{\exp(e_{t,s})}{\sum_{s=1}^n \exp(e_{t,s})}$
      - 根据注意力加权后的LSTM特征：$c_t=\sum_{s=1}^n a_{t,s}h_s$
      - 注意力机制自身的状态更新：$h_t=GRU((embed_{t-1},c_t),h_{t-1})$
      - 根据注意力机制的状态，一层全量网络算出当前（未归一化）预测 $y_t=\mathbf{w}h_t+\mathbf{b}$
      - 而最终用到的结果是归一化的预测 $u_t=\text{softmax}(\mathbf{V}^T h_t)$
  - Adaptive End-to-End Training
    - 训练时，为了稳定，直接使用Ground Truth来训练识别分支。但在测试时，cropping的区域不理想，导致结果变差。
    - 为解决此问题，AET使用了一个**置信阈值**（confidence threshold），并使用**NMS**（非极大值抑制 Non-Maximum Suppression）来提出冗余的检测结果
    - 一开始coordinate convolution module检测到的控制点 $cp$ 是被抑制的，ground truth $cp^*$ 直接替换检测到的控制点，参与训练
    - $rec=\arg\min_{rec^*\in cp^*}\sum_{i=1}^{n}\vert cp^*_{x_i,y_i}-cp_{x_i,y_i}\vert$ 如果rec差太大就用 $cp^*$ 替换掉检测值
  - Text Spotting Quantization（可以理解为模型压缩）
    - 量化text spotting任务，目标是离散化高精度tensor到低比特tensor，而不影响网络性能 
  <center><img height=160 src="/assets/img/papers/model_quantization.png"></center>

- References
  - [ArXiv](https://arxiv.org/abs/2105.03620)
  - [Github](https://github.com/Yuliang-Liu/ABCNet_Chinese)

---

### [PGNet (AAAI 2021)]()
Real-time Arbitrarily-Shaped Text Spotting with Point Gathering Network

- Overview
  - Two stage框架或基于字符的方法上，它们受到非极大抑制 (NMS)、RoI 操作或字符级标注的影响
  - **全卷积点收集网络** (PGNet)：
    - 实时读取任意形状的文本
    - segmentation型单阶段模型
    - 像素级字符分类图是通过PG-CTC损失学习的，从而避免使用字符级的标注
    - 二维空间中收集高级字符分类向量并将它们解码为文本符号，而无需涉及 NMS 和 RoI 操作
    - 图细化模块（graph refinement module = GRM）来优化粗识别并提高端到端性能
    - 可达 46.7 FPS
  - 两种变体
    - PGNet-A = PGNet-Accuracy --> ResNet-50
    - PGNet-E = PGNet-Efficient --> EfficientNet-B0
  - 硬件要求
    - Intel(R) Xeon(R) CPU E5-2620; GPU: NVIDIA TITAN Xp x4; RAM: 64GB
    - 目前只支持PaddlePaddle

- **Framework**
  - 直接从FPN (Feature Pyramid Network) 生成的<img src="https://latex.codecogs.com/svg.image?F_{visual}" /> 学习文本区域的各种信息：
    - 四种feature map大小均为原输入图片的1/4，分别由各自的目标标注 (supervised by the same scale label maps) 做训练
      - 文本字符分类图（pixel-level TCC）：对像素做分类，提供字符级别的信息
      - 文本中心线 (TCL)：获取字符中心点序列
      - 文本方向偏移 (TDO)：获取正确阅读顺序
      - 文本边框偏移 (TBO = boundary offset)：通过多边形修复，检测每个字符实例，针对每个点 $\pi$ 输出上下边界的offset，由此通过中心点算出多边形边界点。

- **PG-CTC解码器 (point gathering - connectionist temporal classification decoder)**
  - 可以序列化高层2D TCC map生成字符分类概率序列，并解码成最终的文字识别结果
  - **免除 字符级别的标注，NMS和RoI**
  - 文字区域中心点序列 $\mathbf{\pi} = \{p_1,p_2,...,p_N\}$ + TCCmap 字符分类概率图 + TCL读取顺序 = 字符概率序列 $P_\pi$
  - $P_\pi = gather(TCC, \pi)$，大小为 $N\times 37$ 的概率向量序列
  - 像素级字符分类图使用点收集 CTC (PG-CTC) 损失进行训练，不需要字符级标注。
  $$L_{PG-CTC}=\sum_{i=1}^MCTC_Loss(P_{\pi_i},L_i)$$
    - P 是检测出的字符分类概率序列
    - L 是转录标签 transcript label（**可以通过word-level标签计算出中心线，并采样获取** $\pi_i$）
  - PG-CTC解码器从测得的概率向量序列 $P_\pi$ 算出 中心点 $\pi$ 的文字记录（transcription）$R_\pi$ 
  $$R_\pi=CTC\_\text{decoder}(P_\pi)$$
<img src="/assets/img/papers/pgnet_structure.png"/>

- **Graph Refinement Module (GRM)** $\rightarrow$ Optional
  - 用来推理文字与其邻居间的关系，以提高正确率
  - 使用图卷积网络（GCN）来实现推理
  - 每个 $\pi$ 中的点都作为一个图中的节点。$F_\text{visual}$ 和 TCC map 被作为节点特征输入，使用**两个独立的GCN进行学习（semantic reasoning graph & visual reasoning graph）**，直到最后采用FC统一处理
  - 关联矩阵（adjacency matrix $A_{ij}$）中元素为 中心点 $p_i$ 和 $p_j$ 间的L2距离 $\mathbf{A}_{ij}=1-D(p_i,p_j)/\max(A)$
  - Semantic Reasoning Graph
    - 注意：$F_s=P_\pi=\text{gather}(\pi,\text{TCC})$, 嵌入获得 $X_s=\text{embed}(F_s)$
  - Visual Reasoning Graph
  $$F_v=\text{gather}(\pi,F_\text{visual})$$
<center><img width=300 src="/assets/img/papers/graph_refinement_module.png"/></center>

- Dataset
  - ICDAR 2015
  - Total-Text

- Comparison
<center><img height=250 src="/assets/img/papers/accuracy_fps_overview.jpg"/></center>
<center><img src="/assets/img/papers/text_detection_recognition_e2e_pgnet.png"/></center>

- 其他算法的缺点
  - TextDragon 和 Mask TextSpotter 强假设文本区域的阅读方向要么是从左到右、要么是从上到下
  - 在实践中免费合成数据并不能完全替代真实数据
  - 在Mask TextSpotter和CharNet中，训练需要字符级的标注，成本太高

- References
  - [ArXiv](https://arxiv.org/abs/2104.05458)
  - [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/16383)
  - [PGNet (知乎)](https://zhuanlan.zhihu.com/p/385115756)
  - [PaperswithCode](https://paperswithcode.com/paper/pgnet-real-time-arbitrarily-shaped-text)