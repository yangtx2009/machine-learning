# [CondLaneNet: a Top-to-down Lane Detection Framework Based on Conditional Convolution](https://github.com/aliyun/conditional-lane-detection)

## Proposal
- Conditional lane detection: resolve the lane instance-level discrimination problem
- Row-wise formulation: no limitation on line shape assumption
- Recurrent Instance Module(RIM): deal with the detection of lane lines with complex topologies such as the dense lines and fork lines

## Methods
- Aim: predict a collection of lanes <img src="https://latex.codecogs.com/svg.image?L&space;=&space;\left\{l_1,&space;l_2,&space;...&space;,&space;l_N&space;\right\}" title="" />. N = total number of lanes
- Each <img src="https://latex.codecogs.com/svg.image?l_k" title="" /> is an ordered set of coordinates (<img src="https://latex.codecogs.com/svg.image?N_k" title="" /> points). <img src="https://latex.codecogs.com/svg.image?N_k" title="" /> = max number of sample points of the kth lane

- Shape Predition
  - **Row-wise Location**: divide the input image into grids of shape Y x X. For each row, we predict the probability (sigmoid) that the lane line appears in each grid.
  - **Vertical Range**: determined by row-wisely predicting whether the lane line passes through the current row (linear layer -> binary classification)
  - **Offset Map**: predict the offset in the horizontal direction near the row-wise location for each row
  - **Shape Description**
- 