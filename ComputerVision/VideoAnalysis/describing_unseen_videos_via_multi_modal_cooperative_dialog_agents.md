# [Describing Unseen Videos via Multi-Modal Cooperative Dialog Agents](https://drive.google.com/file/d/1krU0vcILOr0cZRoyjkpLzPJP-vUtD6tZ/view?usp=drivesdk)

## Overview
- one conversational agent to describe an unseen video based on the dialog and two static frames.
  - Q-BOT: 
    - given two static frames from the beginning and the end of the video, as well as a finite number of opportunities to ask relevant natural language questions before describing the unseen video. Q-BOT只看到一段视频的第一和最后一帧，通过有限数量的问题描述该未看过的完整视频
    - 输入数据有：完整视频，对话历史。最后杨输出视频描述（标题）
    - 最后Q-BOT有能力描述一段不完整的视频
    - 结构
      - a visual module
      - a history encoder
      - a visual LSTM-net
      - a multi-modal attention module
      - a question decoder
      - the final description generator.
  - A-BOT
    - the other agent who has already seen the entire video, assists Q-BOT to accomplish the goal by providing answers to those questions. A-BOT已看过完整视频，它要通过回答问题来协助Q-BOT完成目标
    - 输入数据有：完整音频，完整视频，视频标题（目标），对话历史
    - 结构
      - audio module
      - a visual module
      - a caption encoder
      - a history encoder
      - multi-modal attention module
      - intra-modal attention
      - an answer decoder.
- Cooperative Learning
  - dynamic dialog history update mechanism to help with the knowledge transfer from A-BOT to Q-BOT
  - **use the ground truth dialog as the internal imitation reference for two agents** 训练需要ground truth dialog
   

<p align="center"><img width="450" src="images/describing_unseen_videos_via_multi_modal_cooperative_dialog_agents.png" /></p>

## Knowledges
- multiple modalities of data and subtle reasoning
  - Multi-step reasoning via recurrent dual attention for visual dialog. In: ACL (2019)
  - Compositional attention networks for machine reasoning. In: ICLR (2018)
  - Explore multi-step reasoning in video question answering. In: ACM Multimedia (2018)
  - Stacked attention networks for image question answering. In: CVPR (2016)

## Metrics
- BLEU1-4
- METEOR
- SPICE
- ROUGE_L
- CIDEr

## Dataset
- [AVSD dataset](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge)

## References
- [Github](https://github.com/L-YeZhu/Video-Description-via-Dialog-Agents-ECCV2020)