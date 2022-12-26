# Lane Detection

## Overview
- Segmentation-based methods
- Anchor-based methods
  
  - [End-to-end traffic line detection with line proposal unit 2019](https://ieeexplore.ieee.org/document/8624563)
  - [CurveLane-NAS 2020](https://paperswithcode.com/paper/curvelane-nas-unifying-lane-sensitive)
  - [LaneATT 2020](https://github.com/lucastabelini/LaneATT)
- Row-wise methods
  - [FastDraw: Addressing the long tail of lane detection by adapting a sequential prediction network 2019](https://arxiv.org/abs/1905.04354)
  - [Ultra fast lane detection 2020](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
  - [End-to-end lane marker detection via row-wise classification 2020](https://arxiv.org/abs/2005.08630)
  - 
- Parametric prediction methods (faster but not more accurate)
  - [PolyLaneNet 2020](https://github.com/lucastabelini/PolyLaneNet)
  - [lane shape prediction with transformers (LSTR) 2021](https://github.com/liuruijin17/LSTR)

## Challenges
### > Instance-level discrimination
- **Solution 1**: predict lane points + aggregate the points into lines.
  - Label the lane lines into classes (instance ID) of a fixed number and make a multi-class classification
  - Drawback: hard to assign different points to different lane instances. Only fixed number of lanes can be detected. 

- **Solution 2**: anchor-based methods
  - Drawback: not flexible to predict the line shape due to the fixed shape of the anchor.

### > Detection of lane lines with complex topologies (fork/dense lines)



## Dataset
- MIKKI
- TuSimple
- [CULane](https://xingangpan.github.io/projects/CULane.html)
- [CurveLane](https://github.com/SoulmateB/CurveLanes)

# History
- [Lane detection in driving assistance system 2008](https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.20255)
- Spatial relationship modeling
  - Markov Random Fields (MRF)
    - [MRFNet 2016](mrfnet.md)
  - Conditional Random Fields (CRF)
    - [DenseCRF (NIPS 2011)](https://arxiv.org/abs/1210.5644)
      - https://blog.csdn.net/qq_31347869/article/details/91344524

## Benchmark
- [Papers with Code Task](https://paperswithcode.com/task/lane-detection/latest)
  - the structures are diverse and complex
  - contains various radius of curvature
  - discontinuous dotted lines
- [awesome lane detection](https://github.com/amusi/awesome-lane-detection)