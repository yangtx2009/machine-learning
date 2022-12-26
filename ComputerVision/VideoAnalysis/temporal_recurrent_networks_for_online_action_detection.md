# [Temporal Recurrent Networks for Online Action Detection (TRN)](https://drive.google.com/file/d/1XmDPrrde9hCO-lOeFaSmImaMZe0r6p7c/view?usp=drivesdk)


## Overview
- accumulated historical evidence and predicted future information
- train a network that predicts actions several frames into the future, and then uses that prediction to classify an action in the present 根据历史证据预测未来一段的动作，借此对当前动作做分类
- 对K个可能的动作估计概率分布

## Opinions
- explicitly predicting the future can help to better classify actions in the present.

## Method
![](images/TRN.png)
- 基本架构就是一个RNN Cell，读入当前t时刻的帧<img src="https://latex.codecogs.com/svg.image?I_t" title="" />和上一个时刻的隐藏状态<img src="https://latex.codecogs.com/svg.image?h_{t-1}" title="" />，预测当前的动作类型分布<img src="https://latex.codecogs.com/svg.image?\mathbf{p}_{t}" title="" />
- 内部的结构由三部分组成
  - a temporal decoder: 学习一个特征表达，并预测未来的动作<img src="https://latex.codecogs.com/svg.image?\tilde{p}_{t}^{l_d}" title=""/>
  - a future gate: 收到decoder的所有隐藏状态<img src="https://latex.codecogs.com/svg.image?\tilde{f}_{t}^{l_d}" title=""/>并嵌入这些特征作为未来的context <img src="https://latex.codecogs.com/svg.image?\tilde{x}_t = \text{RELU}(\mathbf{W}_f^T \text{AvgPool}(\tilde{\mathbf{h}}_t)+\mathbf{b}_f)" title=""/>
  - **a spatiotemporal accumulator (STA)**: 根据 过往context <img src="https://latex.codecogs.com/svg.image?h_{t-1}" title=""/>，当前状况 <img src="https://latex.codecogs.com/svg.image?x_t" title=""/> 和预测出来的未来context <img src="https://latex.codecogs.com/svg.image?\tilde{x}_t" title=""/>，预测当前的动作类型分布 <img src="https://latex.codecogs.com/svg.image?\mathbf{p}_t" title=""/>

![](images/TRN2.png)

## Dataset
- [HDD](https://paperswithcode.com/paper/toward-driving-scene-understanding-a-dataset)
- TVSeries
- THUMOS’14

## Reference
- [Scene Understanding](https://paperswithcode.com/task/scene-understanding)