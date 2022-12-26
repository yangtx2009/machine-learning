# [Delving Deeper into the Decoder for Video Captioning](https://drive.google.com/file/d/1xDuOueKyPAL-ZrJPDrWcPv066O3Tl9zK/view?usp=drivesdk)

## Overview
- 目前做video captioning的encoder-decoder framework存在很多问题
  - video caption decoders usually suffer from the serious problem of overfitting. decoder经常过拟合
  - single metric can't reflect the overall performance of video captioning system 单一评价标准不能反映整体性能
  - likely to learn an intersection of the annotations for each video which consists of frequent words and phrases and inclined to forget advanced words and complicated sentence structures 学到的是常见词，而倾向于忘记罕见复杂词

- 优化的方法有（主要是对训练过程的优化建议）
  - a combination of variational dropout and layer normalization is embedded into a recurrent unit to alleviate the problem of overfitting 在一个递归单元中嵌入了variational dropout和layer normalization的组合，以缓解过度拟合的问题
- a new online method is proposed to evaluate the performance of a model on a validation set to find the best checkpoint 使用一个新方法对训练模型做评估，已找到最好的checkpoint
- a professional learning is proposed which uses the strengths of a captioning model and bypasses its weaknesses 优化了训练流程 (teacher forcing)
  1. a model will be trained by optimizing losses computed with training samples equally, which is called teacher forcing or general learning.
  2. n annotations are sampled for the video k
     - Compute and optimize the model by weighted loss function with idxcur, inputs i and human annotation a

## Metrics
- BLEU
- CIDEr
- METEOR
- ROUGE-L

## Dataset
- Microsoft Research Video Description Corpus (MSVD)
- MSR-Video to Text (MSR-VTT) datasets