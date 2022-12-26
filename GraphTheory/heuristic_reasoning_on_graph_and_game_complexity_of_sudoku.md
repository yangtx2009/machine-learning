# [Heuristic Reasoning on Graph and Game Complexity of Sudoku](https://drive.google.com/file/d/1sCMdBtETD2Plo5mDU1M_n2MUQZjtlbFf/view?usp=drivesdk)

## Overview
- 使用一个RNN来遍历sudoku的所有81个node的组合
- 每次RNN更新都输出一个message,一个output
- output输出的是当前位置的数字归属概率
- 输入的x是已知的数字
- node得到已知数字后发布message给其他相邻node
- 其他node综合(加法)收到的所有message并使其推理应该填的数字