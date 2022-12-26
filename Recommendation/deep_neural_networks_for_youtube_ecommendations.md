# Deep Neural Networks for YouTube Recommendations

## Overview
- 候选生成模型+排名模型
- 用户活动历史 作为 条件输入，对百万级别数据进行筛选，选出几百个候选
- ranking对每个筛选出的视频预测出一个score，实际是一个softmax输出的多分类器

## Methods
- [extreme multiclass classification](https://drive.google.com/file/d/1AZFjf5XBIdG7BJamIBacMTyZsIWTkTra/view?usp=drivesdk)
  - 人为对negative class进行取样，降低negative样本数效果比hierarchical softmax好
- scoring scheme sublinear