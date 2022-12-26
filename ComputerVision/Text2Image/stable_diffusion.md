# High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)

![](https://miro.medium.com/max/720/0*rW_y1kjruoT9BSO0.png)
- Perceptual compression model （红色部分）
  - E是一个encoder，将原图压缩为一个隐特征 z。z仍然保留着2D，但是大小比原图小。
  - D为一个decoder，将z还原成目标图片
  - 去除高频，不可察觉的细节
- Latent Diffusion Models （绿色部分）
  - image-specific inductive biases 图像特有的感应偏差
  - 使用单个UNet去噪，该UNet会被用到每个去噪步骤上
  - **UNet输出的不是去噪的图像，而是要去除的噪音**
- Conditioning Mechanisms
  - 依照条件生成图片
  - 在UNet中使用cross-attention mechanism实现条件生成
  - 使用domain specific encoder，将条件y映射到特征 τ_θ，作用在Key和Value上，Query仍然保留来自z_t的信息
  - 条件y可以是
    - BERT-tokenizer+Transformer生成的语言隐特征，用以做text-to-image
    - pixel-level的分类图片，用以做image-to-image
    - 选择一定区域，用以做impainting，缺失补全或遮挡去除

![](https://github.com/yangtx2009/Research/blob/f4602959a284c1c0d8d8006393d5b20f2f61e255/contents/ML/ComputerVision/Text2Image/images/stable-diffusion.png)

- 向量量子化变分自编码机 Vector Quantized VAE (VQ-VAE)
  - 代码中使用了Vector Quantizer
  - VAE的思路是如果强迫潜在表示 z 满足正态分布，那么训练完成后，丢掉编码器，直接从正态分布中抽样得到潜在表示z，使用解码器就可以生成新样本了
  - VQ-VAE最大的特点就是, z的每一维都是离散的整数，以此忽略连续空间带来的过多细节
  - 先有一个codebook, 这个codebook是一个embedding table. 我们在这个table中找到和vector最接近(比如欧氏距离最近)的一个embedding, 用这个embedding的index来代表这个vector
  - embedding table相当于对z空间做聚类，使用紫色表中最近的embedding表示

![](https://vitalab.github.io/article/images/VQ-VAE/Architecture.png)


其他资源
- [github](https://github.com/CompVis/stable-diffusion)
- [VQ-VAE解读](https://zhuanlan.zhihu.com/p/91434658)
