# [Self-Attentive Sentence Embedding](https://drive.google.com/file/d/1fQ3LQvOVzl1WESXL9NK_RUkapDke3fFh/view?usp=drivesdk)

## Overview
- n:句子长度
- d:单词维度
- u:hidden state维度
- da:weight维度
- a与H点积得到固定长度的representation m
- 一个m只关注句子的一部分,通过增加多个(r个)m,也就是M(r*2u),可以得到对句子不同部分的关注

## Analysis
- 优点
  - Self-attention
- 缺点
  - 无法并行, seq2seq
  - 经过太多nonlinearity到hidden state, 难以得到长序列有效信息