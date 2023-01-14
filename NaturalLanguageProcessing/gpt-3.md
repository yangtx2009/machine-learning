# [GPT-3: Language Models are Few-Shot Learners (NIPS 2020)](https://drive.google.com/file/d/1vMkMurPFQYrGednPe8So80hufMKdw92E/view?usp=drivesdk)
## 动机
- 增强大型预训练语言模型在小数据量不可知任务上的性能
scaling up language models greatly improves task-agnostic,
few-shot performance
- GPT-3并没有在GPT-2上有结构上的改动，只是对GPT-2在各种应用上和模型规模上的进一步研究
- 元学习的核心思想在于通过少量的数据寻找一个合适的初始化范围，使得模型能够在有限的数据集上快速拟合，并获得不错的效果。The core idea of meta-learning is to find a suitable initialization range with a small amount of data, which enables the model to be fitted quickly and with good results on a limited data set.


## 方法
- an autoregressive language model (Transformer) with 175 billion
parameters 自回归模型指当前输出y(t)完全基于历史输出y(t-i)，而非x
- 使用以下设置
    - Fine-Tuning (FT)
        - updating the weights of
a pre-trained model by training on a supervised dataset specific to the desired task
        - need for a new large dataset for every task
        - do not fine-tune GPT-3 because the focus is on task-agnostic performance
    - Few-Shot (FS)
        - the model is given a few demonstrations of the task at inference time as conditioning, but no weight updates are allowed
        - 通过学习大量数据集，学习一个泛化模型。之后不需要更新参数，只提供少数上下文例子，即可完成某种特殊任务。
    - One-Shot (1S)
        - 只提供一个示例，以及一个对任务的自然语言描述
    - Zero-Shot (0S)
        - 只提供对任务的自然语言描述
![](https://miro.medium.com/max/720/1*4WVUYA3tJ0wxyYjT5bahCQ.webp)

- 模型
    - same as GPT-2
    - use alternating dense and locally banded sparse attention patterns in the layers of the transformer 在transformer的各层中使用交替密集和局部滑动窗口稀疏注意模式

- 训练过程
    - use a mixture of model parallelism within each matrix multiply and model parallelism across the layers of the network 使用一种混合并行模型 在每个矩阵乘法中使用，在网络的各层中使用模型并行

## 相关概念
- in-context learning
    - No parameter tuning need
    - Only need few examples for downstream tasks
    - [Explanation](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/lectures/lec07.pdf)

## References
- [ChatGPT: The End of Online Exam Integrity? (Arxiv)](https://arxiv.org/abs/2212.09292)
- [OpenAI](https://openai.com/blog/chatgpt/)