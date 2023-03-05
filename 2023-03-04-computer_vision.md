## [detection]({{site.machine-learning}}/ComputerVision/Detection/2023-03-04-detection.md)

- R-CNN 2014
  - two-stages: pretrained CNN (one forward pass for each object proposal) + SVM
- [Fast R-CNN 2015](/ComputerVision/Detection/fast-rcnn.md)
  - SPPnet (one forward pass for all proposals) + RoI pooling layer
- [Faster R-CNN 2015](ComputerVision/Detection/faster-rcnn.md) <span> &#128293;</span>
  - region proposal network (RPN) to replace RoI pooling
- SSD 2016
- [YOLO (CVPR 2016)](ComputerVision/Detection/yolo.md)
- [YOLOv2 (CVPR 2017)](ComputerVision/Detection/yolo.md)
- [RetinaNet 2017](ComputerVision/Detection/RetinaNet.md)
  - Focal loss (solve data imbalance) + Feature Pyramid Network + RPN
- [<span>YOLOv3 2019 &#128293;</span>](ComputerVision/Detection/yolo.md)
- [YOLOv4 2020](ComputerVision/Detection/yolo.md)
- RelationNet++ 2020
- DETR 2020
- UP-DETR 2020

## [one/few shot object detection](ComputerVision/OneFewShot/one-or-few-shot-object-detection.md)
- [OS2D (ECCV 2020)]()
- [One-Shot Object Detection without Fine-Tuning (2020)]()
- [Quasi-Dense Similarity Learning for Multiple Object Tracking (CVPR 2021)]()
- [Meta Faster R-CNN (AAAI 2022)]()
- [Semantic-Aligned Fusion Transformer for One-Shot Object Detection (CVPR 2022)]()
- [One-Shot General Object Localization (arxiv 2022)]()
- [Balanced and Hierarchical Relation Learning for One-Shot Object Detection (CVPR 2022)]()
- [Simple Open-Vocabulary Object Detection with Vision Transformers (ECCV 2022)]()

## **segmentation**
- Regional proposal based
  - [<span>Mask R-CNN (ICCV 2017) &#128293;</span>](ComputerVision/Segmentation/mask_rcnn.md)
    - Binary ROI mask
    - RoIAlign
  - [LaneNet (IEEE IV 2018)](https://arxiv.org/abs/1802.05591)
  - [Mask Scoring R-CNN (CVPR 2019)](https://ieeexplore.ieee.org/document/8953609)

- RNN based
  - ReSeg
  - MDRNNs

- Upsampling + Deconvolution
  - [FCN (CVPR 2015 & TPAMI 2017)](https://arxiv.org/abs/1605.06211v1)
  - SetNet
  - [U-net (MICCAI 2015)](https://drive.google.com/file/d/1GIOJgIe1BzChxoIWJyZq4G7EQOAE4OPY/view?usp=drivesdk)
  - [FastFCN 2019](https://drive.google.com/file/d/1wIo5dLL_Sn2Bxlo4cGx4YacONw1mJU6N/view?usp=drivesdk)

- CRF/MRF
  - [DeepLab (ICLR 2015 & ICCV 2015)](https://drive.google.com/file/d/1X0S9WRAzMTG0hQbaysdR60prPdhPRf9E/view?usp=drivesdk)
    - Fully connected CRF
    - Atrous (Dilated) Convolution
  - [CRF Meet Deep Neural Networks for Semantic Segmentation 2018](https://drive.google.com/file/d/1oQTA8xbPvoBV0IQbOnnBsKLZVocJieQg/view?usp=drivesdk)

- Gated-SCNN 2019

- Interactive
  - [Interactive Image Segmentation With First Click Attention (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Interactive_Image_Segmentation_With_First_Click_Attention_CVPR_2020_paper.pdf)
  - [F-BRS (CVPR 2020)](https://ieeexplore.ieee.org/document/9156403)
  - [Interactive Object Segmentation With Inside-Outside Guidance (CVPR 2020)](https://ieeexplore.ieee.org/document/9157733)

- Moving Object
  - [An End-to-End Edge Aggregation Network for Moving Object Segmentation (CVPR 2020)](https://ieeexplore.ieee.org/document/9156405)

- Transparent Object
  - [Deep Polarization Cues for Transparent Object Segmentation (CVPR 2020)](https://ieeexplore.ieee.org/document/9156916)

- Referring 转介
  - [PhraseCut (CVPR 2020)](https://ieeexplore.ieee.org/document/9157191)
  - [Referring Image Segmentation via Cross-Modal Progressive Comprehension (CVPR 2020)](https://ieeexplore.ieee.org/document/9156414)
- Multi-Object Tracking
  - [Learning Multi-Object Tracking and Segmentation From Automatic Annotations (CVPR 2020)](https://ieeexplore.ieee.org/document/9157138)

## [**lane detection**](ComputerVision/LaneDetection/lane_detection.md)
- Segmentation-based
  - [SCNN (ISCA 2017)](ComputerVision/LaneDetection/scnn.md)
  - [ENet-SAD (ICCV 2019)](ComputerVision/LaneDetection/enet_sad.md)
  - CurveLanes-NAS
  - LaneNet 2019
  - RESA 2020
  - SUPER 2020
  - ERFNet-IntRA-KD 2020
- Row-wise
  - E2E-LMD
  - IntRA-KD
- Other approaches

## **generation**
- image generation
  - Parzen window-based log-likelihood
  - [GAN 2014](https://arxiv.org/abs/1406.2661)
- textual descriptions
  - [<span>Neural Image Caption Generation with Visual Attention &#x1F448;</span>](https://arxiv.org/abs/1502.03044)

## [**ocr**](./ComputerVision/OCR/ocr.md)
- [Spatial Transformer Networks (NIPS 2015)](ComputerVision/OCR/spatial_transformer_networks.md)
- [CTPN (ECCV 2016)](ComputerVision/OCR/detecting_text_in_natural_image_with_connectionist_text_proposal_network.md) - Text Detection
- [CRNN+CTC (TPAMI 2017)](ComputerVision/OCR/crnn_ctc.md) - Text Recognition
- [EAST (CVPR 2017)](ComputerVision/OCR/EAST-an_efficient_and_accurate_scene_text_detector.md) - Text Detection
- [ASTER (TPAMI 2018)](ComputerVision/OCR/ASTER.md) - Text Recognition
- [FOTS (CVPR 2018) &#128293;](ComputerVision/OCR/fots_fast_oriented_text_spotting_with_a_unified_network.md) - **End2End** Linux Only 
- [2D atttention (ICCV 2019)](ComputerVision/OCR/towards_end-to-end_text_spotting_in_natural_scenes.md) - **End2End**
- [Character Region Awareness for Text Detection (CVPR 2019)]()
- [Convolutional Character Networks (ICCV 2019)](ComputerVision/OCR/convolutional_character_networks.md)
- [Aggregation Cross-Entropy (CVPR 2019)](ComputerVision/OCR/aggregation_cross-entropy_for_sequence_recognition.md)
- [Mask TextSpotter (TPAMI 2019)](ComputerVision/OCR/mask-textspotter.md) - **End2End**
- [ESIR (CVPR 2019)](ComputerVision/OCR/end-to-end_scene_text_recognition_via_iterative_image_rectification.md) - Text Recognition
- [SATRN (CVPR 2020)](ComputerVision/OCR/on_recognizing_texts_of_arbitrary_shapes_with_2D_self-attention.md) - Text Recognition
- [DAN (AAAI 2020)](ComputerVision/OCR/decoupled_attention_network_for_text_recognition.md) - Text Recognition
- [Yet Another Text Recognizer (ArXiv 2021)](ComputerVision/OCR/why_you_should_try_the_real_data_for_the_scene_text_recognition.md) - Text Recognition
- [ABCNet (CVPR 2020)](ComputerVision/OCR/ABCNet_real-time_scene_text_spotting_with_adaptive_bezier-curve_network.md) - **End2End** Linux Only
- [ABCNet v2 (CVPR 2021) &#128293;](ComputerVision/OCR/abcnet_v2.md) - **End2End** Linux Only
- [PGNet (AAAI 2021) &#128293;](ComputerVision/OCR/pgnet.md) - **End2End**


## **video analysis**
- [Video Description, Video Reasoning](ComputerVision/VideoAnalysis/video_description.md)
  - [Describing Videos by Exploiting Temporal Structure (ICCV 2015)](ComputerVision/VideoAnalysis/describing_videos_by_exploiting_temporal_structure.md)
  - [Localizing Moments in Video with Natural Language (ICCV 2017)]()
  - [<span>Grounded Video Description (CVPR 2019) &#128293;</span>](ComputerVision/VideoAnalysis/grounded_video_description.md)
  - [Video Relationship Reasoning Using Gated Spatio-Temporal Energy Graph (GSTEG, CVPR 2019)](video_relationship_reasoning_using_gated_spatio-temporal_energy_graph.md) - CRF
  - [Adversarial Inference for Multi-Sentence Video Description (CVPR 2019)](ComputerVision/VideoAnalysis/adversarial_inference_for_multi-sentence_video_description.md) - adversarial inference
  - [Delving Deeper into the Decoder for Video Captioning (ECAI 2020)](ComputerVision/VideoAnalysis/delving_deeper_into_the_decoder_for_video_captioning.md)
  - [Identity-Aware Multi-Sentence Video Description (ECCV 2020)](ComputerVision/VideoAnalysis/identity-aware_multi-sentence_video_description.md)
  - [Describing Unseen Videos via Multi-Modal Cooperative Dialog Agents (ECCV 2020)](ComputerVision/VideoAnalysis/describing_unseen_videos_via_multi_modal_cooperative_dialog_agents.md)
  - [<span>Exploiting Visual Semantic Reasoning for Video-Text Retrieval (IJCAL 2020) &#128293;</span>](ComputerVision/VideoAnalysis/exploiting_visual_semantic_reasoning_for_video-text_retrieval.md)
  - [Fine-Grained Video-Text Retrieval With Hierarchical Graph Reasoning (CVPR 2020)](ComputerVision/VideoAnalysis/fine-grained_video-text_retrieval_with_hierarchical_graph_reasoning.md)
  - [A Hierarchical Reasoning Graph Neural Network for The Automatic Scoring of Answer Transcriptions in Video Job Interviews (ArXiv 2020)](ComputerVision/VideoAnalysis/a_hierarchical_reasoning_graph_neural_network_for_the_automatic_scoring_of_answer_transcriptions_in_video_job_interviews.md)
  - [Reasoning with Heterogeneous Graph Alignment for Video Question Answering (AAAI 2020)](ComputerVision/VideoAnalysis/reasoning_with_heterogeneous_graph_alignment_for_video_question_answering.md)
  
- [Temporal Action Detection](ComputerVision/VideoAnalysis/temporal_action_detection.md)
  - frame-based
    - [CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos (CVPR 2017)]()
    - [TAG: Temporal Action Detection with Structured Segment Networks (ICCV 2017)]()
    - [I3D (CVPR)]( 2017https://zhuanlan.zhihu.com/p/84956905)
    - [SlowFastNet (ICCV 2019)](https://arxiv.org/pdf/1812.03982.pdf)
  - proposal-based
    - [SCNN Segment-CNN (CVPR 2016)](ComputerVision/VideoAnalysis/temporal_action_localization_in_untrimmed_videos_via_multi-stage_cnns.md)
    - [TURN TAP (ICCV 2017)](ComputerVision/VideoAnalysis/TURN-TAP.md)
    - [CBR: Cascaded Boundary Regression (BMVC 2017)](ComputerVision/VideoAnalysis/CBR_cascaded_boundary_regression.md)
  - attention-based
    - [TimeSformer](https://zhuanlan.zhihu.com/p/357848386)
  - self-supervised
    - [Visual&CBT](https://zhuanlan.zhihu.com/p/250477141)
  - online action detection (异常事件监测)
    - [Online Action Detection (ECCV 2016)](ComputerVision/VideoAnalysis/online_action_detection.md)
    - [RED: Reinforced Encoder-Decoder Networksfor Action Anticipation (BMVC 2017)](ComputerVision/VideoAnalysis/RED.md)
    - [Online Action Detection in Untrimmed, Streaming Videos (ECCV 2018)]()
    - [Temporal Recurrent Networks for Online Action Detection (ICCV 2019)](ComputerVision/VideoAnalysis/temporal_recurrent_networks_for_online_action_detection.md)

  - multiple actors
    - [Action Understandingwith Multiple Classes of Actors](https://drive.google.com/file/d/1ta4UmPSjyjzC7mJCp2AldvMs-uh42NMU/view?usp=drivesdk)
    - [SSA2D](https://www.crcv.ucf.edu/wp-content/uploads/2020/12/Projects_Single-shot-actor-action-detection-in-videos.pdf)

## 3D Reconstruction
- [<span>NeRF (ECCV 2020) &#128293;</span>](https://www.matthewtancik.com/nerf)
<!-- [link]({{ site.url }}/assets/html/machine_learning.html) -->
