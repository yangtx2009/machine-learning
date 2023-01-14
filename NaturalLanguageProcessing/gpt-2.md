# [Language Models are Unsupervised Multitask Learners (2019)](https://drive.google.com/file/d/1qG1Y2yQEeIEt4Ol3At3CnF2AxndyDDGU/view?usp=drivesdk)

## 动机
- 表明语言模型在没有现式监督学习下，可以学习多任务 language
models begin to learn these tasks without any explicit supervision
- 并没有对GPT-1的网络进行过多的结构的创新与设计，只是使用了更多的网络参数和更大的数据集

## 方法
- 对语言建模通常被描述为对一组文字（变化长度的字符串）的无监督分布估计
- 由于一个通用的系统应该能够执行许多不同的任务，即使是对于相同的输入，它不仅应该以输入为条件，还应该以要执行的任务为条件 p(output|input; task)。 Since a general system should be able to perform many different tasks, even for the same input, it should condition not only on the input but also on the task to be performed.
- 预处理
    - lower casing
    - tokenization
    - out-of-vocabulary tokens which restrict the space of model-able strings
    - Byte Pair Encoding (BPE)

![https://zhuanlan.zhihu.com/p/78153185](https://pic2.zhimg.com/v2-d909e1d04bd94fba1975120f1f041815_b.webp)

## References
- [PapersWithCode](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask)