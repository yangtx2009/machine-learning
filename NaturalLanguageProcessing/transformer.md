# [Transformer](https://drive.google.com/file/d/1qeUtnjPJsM8CtqlCnNTh0EJYwqcZZDO2/view?usp=drivesdk)

- 使用不同周期的正弦函数组合对位置做encoding
- Q和K的点乘共同生成权重,Q代表当前单词,K表示比较单词, 再与V相乘
- Transformer工作流程是
  - input全部输入encoder得到输出特征
  - encoder的输出参与decoder中部的输出
  - decoder先prefix,并计算attention输出, 杂糅encoder输出并预测下个词,再将其作为输入,类似seq2seq
- self-attention中对所有输入进行杂糅,feed forward NN是对特征分开处理的
- Encoder输出m,则Encoder-Decoder-Attention block只通过m得到K,V,而Q来源于output(shifted right)的attention结果
- 与seq2seq相比:
  - Pros
    - lower computational complexity per layer
    - can be parallelized (with respect to RNN)
    - Path length between long-range dependencies in the network
  - Cons:
    - hard to accomplish some simple tasks like string copy
    - not computationally unverisal
- Solution: Universal Transformers


## Reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformer模型原理详解](https://zhuanlan.zhihu.com/p/44121378)
- [Attention Is All You Need (Yannic Kilcher) Video](https://www.bilibili.com/video/BV1cW411V7A7/?spm_id_from=trigger_reload)
- [NLP中的Attention原理和源码解析](https://zhuanlan.zhihu.com/p/43493999)