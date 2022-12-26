# Temporal Action Detection

## Institues & Researchers
| Team | Member |
|------|--------|
| KU Leuven - ESAT - PSI | Roeland De Geest, Amir Ghodrati, Tinne Tuytelaars |
| University of Amsterdam  | Efstratios Gavves, Zhenyang Li, Cees Snoek|
|University of Southern California|Jiyang Gao, Zhenheng Yang, Kan Chen, Ram Nevatia|
|Google Research|Chen Sun|
|Indiana University|Mingze Xu, David J. Crandall|
|University of Maryland|Mingfei Gao, Larry S. Davis|
|Honda Research Institute, USA|Yi-Ting Chen|

## Opinions
- [计算机视觉中video understanding领域有什么研究方向和比较重要的成果？](https://www.zhihu.com/question/64021205/answer/866060224)
  - detection的目的是找到boundary 
  - 对于temporal detection，也许最重要的不是找到非常precise的start/end boundary，而是找到一个在动作内部的时间点
  - 这个领域没有什么特别好的数据集，THUMOS比较小，Activitynet的temporal boundary标的不太好，经常会有长度达到1分钟的action标注

## Keywords
- untrimmed videos
- action detection
- Temporal Action Proposal (TAP) generation = temporal window
- online action detection problem: requires us to process each frame as soon as it arrives, without accessing any future information.
- video summarization & video synopsis
- Two-Stream
  - Video = Appearance + Motion
  - Spatial stream ConvNet + Temporal stream ConvNet

References
- [视频理解近期研究进展](https://zhuanlan.zhihu.com/p/36330561)
- [万字长文漫谈视频理解](https://zhuanlan.zhihu.com/p/158702087)
- [<span>综述 | MIT提出视频理解/行为识别：全面调研（2004-2020）&#128293;</span>](https://zhuanlan.zhihu.com/p/282081673)
- [工业界视频理解解决方案大汇总](https://zhuanlan.zhihu.com/p/331660909)
- [管中窥”视频“，”理解“一斑 —— 视频理解概览](https://zhuanlan.zhihu.com/p/346985087)
- [PySlowFast提供视频理解基线(baseline)模型，还提供了当今前沿的视频理解算法复现](https://zhuanlan.zhihu.com/p/101606964)
- [MMAction2: 新一代视频理解工具箱](https://zhuanlan.zhihu.com/p/347705276)
- [cs231n Video understanding](http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds08.pdf)