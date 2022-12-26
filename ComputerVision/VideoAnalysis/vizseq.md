# [VizSeq: A Visual Analysis Toolkit for Text Generation Tasks](https://drive.google.com/file/d/1SzGIPCtG0rs66-BCdJowL6HfEuqChAVq/view?usp=drivesdk)

## Overiew

Instead of using BLEU or ROUGE (abstract numbers and not perfectly aligned with human assessment), **VizSeq** is a visual analysis toolkit for instance-level and corpus-level system evaluation on a wide variety of text generation tasks.

VizSeq不是使用BLEU或ROUGE（抽象的数字，与人的评估不完全一致），而是一个视觉分析工具包，用于对各种文本生成任务进行实例级和语料库级系统评估。

Related metrics in the toolkit:
- N-gram-based metrics
  - [BLEU (bilingual evaluation understudy 双语评估研究)](https://www.aclweb.org/anthology/P02-1040.pdf)
  - [NIST (National Institute of Standards and Technology)](https://dl.acm.org/doi/10.5555/1289189.1289273)
  - [METEOR](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Banerjee-Lavie-2005-METEOR.pdf)
  - [TER (Translation Edit Rate)](http://www.cs.umd.edu/~snover/pub/amta06/ter_amta.pdf)
  - [RIBES (Rank-based Intuitive Bilingual Evaluation Score)](http://www.kecl.ntt.co.jp/icl/lirg/ribes/)
  - [GLEU (Google-BLEU)](https://arxiv.org/abs/1609.08144)
  - [ROUGE](https://www.aclweb.org/anthology/W04-1013/)
  - [CIDEr (Consensus-based image description evaluation)](https://arxiv.org/abs/1411.5726)
- Embedding-based metrics
  - [BERTScore](https://arxiv.org/abs/1904.09675)
  - [LASER (Language-Agnostic SEntence Representations)](https://arxiv.org/abs/1812.10464)
    - https://github.com/facebookresearch/LASER

## Dataset
- [WMT14 English-German](https://paperswithcode.com/sota/machine-translation-on-wmt2014-english-german)
- [Gigaword](https://www.tensorflow.org/datasets/catalog/gigaword)
- [COCO captioning 2015](https://cocodataset.org/#captions-2015)
- [COCO captioning 2015 Tensorflow](https://www.tensorflow.org/datasets/catalog/coco_captions)
- [WMT16 multimodal machine translation task](https://www.statmt.org/wmt16/multimodal-task.html)
- [Multilingual machine translation on TED talks
dataset](http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/)
-[IWSLT17 English-German speech translation](https://isl.anthropomatik.kit.edu/downloads/IWSLT2017SystemPaper.pdf)
- [YouCook2](http://youcook2.eecs.umich.edu/)

## References
- [VizSeq: A Visual Analysis Toolkit for Text Generation Tasks](https://arxiv.org/pdf/1812.06587v2.pdf)
- [facebookresearch/vizseq](https://github.com/facebookresearch/vizseq)
- [PapersWithCode](https://paperswithcode.com/paper/vizseq-a-visual-analysis-toolkit-for-text)