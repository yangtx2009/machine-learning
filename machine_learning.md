# Machine Learning
## General 
- [Resnet 2015](https://arxiv.org/abs/1512.03385)
  - residual learning
- [Layer normalization 2016](https://arxiv.org/abs/1607.06450)
  - layer normalization
- [ResNeSt 2020](https://drive.google.com/file/d/1QlGDm0RTg7SJFegwRIll3Egnvczp-mWo/view?usp=drivesdk) (split-attention)

---
## [Computer Vision](ComputerVision/computer_vision.md)
### [detection](ComputerVision/Detection/detection.md)
- R-CNN 2014
  - two-stages: pretrained CNN (one forward pass for each object proposal) + SVM
- [Fast R-CNN 2015](ComputerVision/Detection/fast-rcnn.md)
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

### [one/few shot object detection](ComputerVision/OneFewShot/one-or-few-shot-object-detection.md)
- [OS2D (ECCV 2020)]()
- [One-Shot Object Detection without Fine-Tuning (2020)]()
- [Quasi-Dense Similarity Learning for Multiple Object Tracking (CVPR 2021)]()
- [Meta Faster R-CNN (AAAI 2022)]()
- [Semantic-Aligned Fusion Transformer for One-Shot Object Detection (CVPR 2022)]()
- [One-Shot General Object Localization (arxiv 2022)]()
- [Balanced and Hierarchical Relation Learning for One-Shot Object Detection (CVPR 2022)]()
- [Simple Open-Vocabulary Object Detection with Vision Transformers (ECCV 2022)]()

### **segmentation**
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

### [**lane detection**](ComputerVision/LaneDetection/lane_detection.md)
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

### **generation**
- image generation
  - Parzen window-based log-likelihood
  - [GAN 2014](https://arxiv.org/abs/1406.2661)
- textual descriptions
  - [<span>Neural Image Caption Generation with Visual Attention &#x1F448;</span>](https://arxiv.org/abs/1502.03044)

### [**ocr**](./ComputerVision/OCR/ocr.md)
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


### **video analysis**
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

 
- 3D Reconstruction
  - [<span>NeRF (ECCV 2020) &#128293;</span>](https://www.matthewtancik.com/nerf)
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
---
## [Natural Language Processing](NaturalLanguageProcessing/nature_language_processing.md)
- traditional methods
  - [word2vec (‎Google 2013)](NaturalLanguageProcessing/word2vec.md)
- general
  - [GPT-2/RWC (2019)](NaturalLanguageProcessing/gpt-2.md)
  - [GPT-3 (NIPS 2020)](NaturalLanguageProcessing/gpt-3.md)
- full sentence translation
  - seq2seq
  - [A Neural Probabilistic Language Model (JMRL 2003)](NaturalLanguageProcessing/a_neural_probabilistic_language_model.md)
    - word embeddings
  - [A Structured Self-Attentive Sentence Embedding (ICLR 2017)](NaturalLanguageProcessing/self_attentive_sentence_embedding.md)
    - Self-Attention
  - [Transformer (NIPS 2017)](NaturalLanguageProcessing/transformer.md)
    - Multi-Head Self-Attention
    - Positional Encoding
    - Scaled Dot-Product Attention
  - [Universal transformer (ICLR 2019)](https://arxiv.org/abs/1807.03819)
    - Transition Function
    - Recurrent Mechanism
  - [<span>BERT (Arxiv 2018) &#x1F448;</span>](NaturalLanguageProcessing/bert.md)
- Simultaneous Translation
  - [<span>STACL (ACL 2019) &#x1F448;</span>](https://drive.google.com/file/d/14_1-FOfAf-HZv-y1AHwKpGPpytfZlFcR/view?usp=drivesdk)
- Machine Translation
  - [Continuous space language models (CSL 2007)](https://www.sciencedirect.com/science/article/abs/pii/S0885230806000325)
  - [Statistical Language Models Based on Neural Networks 2012](https://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)

- Speech Recognition
  - [Whisper 2022](NaturalLanguageProcessing/whisper.md)
  - 
- Dialogue Generation
  

---
## [Graph Theory](GraphTheory/graph_theory.md) 
- traditional methods
  - LINE  
  - TADW
- [graph embedding](GraphTheory/graph_embedding.md)
  - [word2vec (ArXiv 2013)](GraphTheory/word2vec.md) - SkipGram
  - [DeepWalk (KDD 2014)](GraphTheory/deepwalk.md) - graph + word2vec
  - [node2vec (KDD 2016)](GraphTheory/node2vec.md)
  - [item2vec (MLSP 2016)](GraphTheory/item2vec.md)
  - [Cross-Modality Attention with Semantic Graph Embedding for Multi-Label Classification (AAAI 2020)]()
  - [GraphZoom (ICLR 2020)]()
  
- [graph neural networks](GraphTheory/graph_neural_network.md)
  - convolutional network
    - spectral
      - [Spectral networks and locally connected networks on graphs (ICLR 2014, 最早的频谱图神经网络)](GraphTheory/spectral_networks_and_deep_locally_connected_networks_on_graphs.md)
      - [Deep convolutional networks on graph-structured data (CoRR 2015)](GraphTheory/deep_convolutional_networks_on_graph-structured_data.md)
      - [Convolutional Neural Networks on Graphswith Fast Localized Spectral Filtering (Chebyshev expansion, NIPS 2016)](GraphTheory/convolutional_neural_networks_on_graphs_with_fast_localized_spectral_filtering.md)
      - [<span>GCN (ICLR 2017) &#128293;</span>](GraphTheory/semi-supervised_classification_with_graph_convolutional_networks.md)
      - AGCN
      - GGP
    - spatial
      - [Convolutional networks on graphs for learning molecular fingerprints (NIPS 2015)](GraphTheory/convolutional_networks_on_graphs_for_learning_molecular_fingerprints.md)
      - [Diffusion-convolutional neural networks (IANIPS 2016)](GraphTheory/diffusion-convolutional_neural_networks.md)
      - Neural FPs
      - PATCHY-SAN
      - MoNet 2017
      - [GraphSAGE (NIPS 2017)](GraphTheory/GraphSAGE.md)
      - SACNN 2018
      - DCNN
  - attention-based network
    - [NLNN]()
    - [One-Shot Imitation Learning (neighborhood attention, NIPS 2017)](GraphTheory/one-shot_imitation_learning.md)
    - [<span>GAT (self-attention, ICLR 2018) &#128293;</span>](GraphTheory/graph_attention_networks.md)
    
  - spatial-temporal graph
  - hierarchical graph
    - [PairNorm (ICLR 2020)](GraphTheory/pairnorm.md)
    - [Subgraph Neural Networks (NIPS 2020)](GraphTheory/subgraph_neural_networks.md)
  
  - relational reasoning
    - [A simple neural network module for relational reasoning (NIPS 2017)](GraphTheory/a_simple_neural_network_module_for_relational_reasoning.md)
    - [Relational-GCN (ESWC 2018)](GraphTheory/modeling_relational_data_with_graph_convolutional_networks.md)
    - [VAIN (NIPS 2017)](GraphTheory/VAIN-attentional_multi-agent_predictive_modeling.md)
    - [CompGCN (ICLR 2020)](GraphTheory/composition-based_multi-relational_graph_convolutional_networks.md)
    - [Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning (ArXiv 2021)](GraphTheory/temporal_knowledge_graph_reasoning_based_on_evolutional_representation_learning.md)

  - image segmentation
    - [AGNN (ICCV 2019)](GraphTheory/zero-shot_video_object_segmentation_via_attentive_graph_neural_networks.md) [![](contents/GitHub.png)](https://github.com/carrierlxk/AGNN)

  - [Heterogeneity 异质性](GraphTheory/heterogeneous_graph_embedding.md)
    - [metapath2vec (KDD 2017)](GraphTheory/metapath2vec.md)
    - [Hindroid (KDD 2017)](hindroid-an-intelligent_android_malware_detection_system_based_on_structured.md)
    - [metagraph2vec (PAKDD 2018)](GraphTheory/metagraph2vec.md)
    - [Heterogeneous graph neural network (KDD 2919)](GraphTheory/heterogeneous_graph_neural_network.md)
    - [Heterogeneous graph attention network (HAN, WWW 2019)](GraphTheory/heterogeneous_graph_attention_network.md)
    - [GATNE (KDD 2019)](GraphTheory/gatne.md)

---
## [Outlier Detection](OutlierDetection/outlier_detection.md)
- One class classification / Out-of-Distribution
- Open set/world recognition
  - [Towards Open World Object Detection (CVPR 2021)](OutlierDetection/towards_open_world_object_detAection.md)
  - [OW-DETR (arxiv 2021)](OutlierDetection/ow-detr.md)
  - [Open-world Object Detection and Tracking (cmu 2021)](OutlierDetection/open-world_object_detection_and_tracking.md)
  - [Revisiting Open World Object Detection (arXiv 2022)](OutlierDetection/re-owod.md)
  - [VOS (ICLR 2022): Learning What You Don’t Know By Virtual Outlier Synthesis]()

---
## [Point Cloud](PointCloud/point_cloud.md)
- Detection
  - [PointNet 2017](PointCloud/point_net.md)
    - classification (feature extraction+max pooling) + segmentation network (global+point feature concatenation)
  - [PointNet++ 2017]()
  - [VoxelNet 2018](PointCloud/voxelnet.md)
    - consider voxel in information aggregation -> only process voxel feature
  - [PointPillars 2019]()
  - [Multi-Level Context VoteNet 2020](PointCloud/mlcvnet.md)
  - [PV-RCNN++ 2022]()
  - [BADet 2022](PointCloud/BADet.md)

- Segmentation
  - [SqueezeSeg 2017](https://arxiv.org/abs/1710.07368)
  - [PointSeg 2018](https://arxiv.org/abs/1807.06288)
  - [Cylinder3D 2020](https://arxiv.org/pdf/2008.01550.pdf)

- Sensor Fusion
  - [SECOND 2018](PointCloud/second.md)
  - [Frustum PointNets 2018](PointCloud/frustum_pointnets.md)
  - [PointFusion 2018](PointCloud/pointfusion.md)
  - [CLOCs 2020: Camera-LiDAR Object Candidates Fusion for 3D Object Detection](PointCloud/CLOCs.md)
  - [Frustum-PointPillars 2021](PointCloud/frustum_pointpillars.md)
    - frustum probability mask (from image) + pointnet (bird-view 2D grid)
  - [Object Detection in 3D Point Clouds via Local Correlation-Aware Point Embedding 2021](PointCloud/object_detection_in_3D_point_clouds_via_local_correlation-aware_point_embedding.md)
  - [AutoAlign 2022](PointCloud/autoalign.md): attention + Cross-modal Feature Interaction

- 2D Image -> 3D Object Detection
  - [ImVoxelNet (WACV 2022)](PointCloud/imvoxelnet.md)

---
## [Recommendation](Recommendation/recommendation.md)
- [LIRA 1998](Recommendation/lira.md)
- [SimRank 2002](https://drive.google.com/file/d/16bSJlpzmGmxAgh-U_zMyl1QJvqhcY_vO/view?usp=drivesdk)
- [Matrix Factorization 2009](https://drive.google.com/file/d/1zPbx8cq_pOqljk3xCmJ0HsafqK7dFP6H/view?usp=drivesdk)
- [WSABIE rank loss 2011](https://drive.google.com/file/d/1LB-iSCGi0UDCZBsDgZpFA-4P9U8NJ8ro/view?usp=drivesdk)
- [XGBoost 2016](https://drive.google.com/file/d/1FTxR2qfzXrP2dui53DGM19_fvEUh9BVj/view?usp=drivesdk)
- [Deep Neural Networks for YouTube Recommendations 2016](Recommendation/deep_neural_networks_for_youtube_ecommendations.md)
- [DLRM 2019](Recommendation/dlrm.md)
- [Two-tower model 2020](https://drive.google.com/file/d/1yLhp2yUVHHmbqtWOxVdLUGoTSW4Odjrm/view?usp=drivesdk)


---
## [Unsupervised Learning](UnsupervisedLearning/unsupervised_learning.md)

|         | With Teacher | Without Teacher |
|---------|--------------|-----------------|
| **Active**  | Reinforcement Learning/Active Learning | Intrinsic Motivation/Exploration |
| **Passive** | Supervised Learning | Unsupervised Learning |

#### [active learning](UnsupervisedLearning/active_learning.md)

#### [representation learning](UnsupervisedLearning/representation_learning.md)
- Colorization 2016
- [Cycle-GAN (ICCV 2017)](https://arxiv.org/abs/1703.10593)
- Unsupervised word translation 2018
- Deep clustering 2018
- [XLM-R 2020](https://drive.google.com/file/d/1FdfNcJDI0Y4QbVRuW79-c9iSvd8VErGp/view)
- [TabNet 2020](UnsupervisedLearning/tabnet.md)
- [On mutual infomration maximization for representation learning 2020](UnsupervisedLearning/on_mutual_infomration_maximization_for_representation_learning.md)


#### autoregressive networks
- WaveNets 2016
- PixelRNN 2016


---
## Transfer Learning
- [Audio Spoofing Verification using Deep Convolutional Neural Networks by Transfer Learning (CoRR 2020)]()

---
## [Domain Adaptation](DomainAdaption\domain_adaption.md) 

---
## Meta-Learning

---
## Continual Learning


---
## Data Augmentation


---
## Concepts
- [data contamination](NaturalLanguageProcessing/gpt-3.md): training dataset can potentially include content from test datasets
- [sentiment analysis](NaturalLanguageProcessing/gpt-2.md): 情绪分析