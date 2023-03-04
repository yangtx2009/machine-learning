# Machine Learning
## General 
- [Resnet 2015](https://arxiv.org/abs/1512.03385)
  - residual learning
- [Layer normalization 2016](https://arxiv.org/abs/1607.06450)
  - layer normalization
- [ResNeSt 2020](https://drive.google.com/file/d/1QlGDm0RTg7SJFegwRIll3Egnvczp-mWo/view?usp=drivesdk) (split-attention)

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