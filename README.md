


<img width="2800" height="800" alt="sample_1 kopyası" src="https://github.com/user-attachments/assets/55619f8b-8d67-4735-8326-70b612e148d1" />


U-Net Semantic Segmentation Framework
This project provides a complete implementation of the U-Net architecture for semantic segmentation using Keras and TensorFlow. The system is designed to perform precise pixel-level classification, allowing for the isolation of specific objects from their backgrounds within digital images.

Project Overview
The core objective of this work is to demonstrate a reliable pipeline for binary semantic segmentation. By classifying every individual pixel, the model creates a detailed map of the input image, distinguishing between the target subject and the surrounding environment.

Core Components

Integrated Data Pipeline: Specialized loaders for synchronizing image and mask pairs.

Structured Architecture: A classic U-Net implementation featuring symmetrical encoding and decoding paths.

Binary Classification: A sigmoid-based output layer optimized for two-class segmentation tasks.

Visualization Tools: Functions to overlay predicted masks onto original images for qualitative analysis.

Model Architecture
The U-Net design is centered around a "contracting" path to capture context and a "symmetric expanding" path to enable precise localization. This is achieved through the following stages:

Encoder (Downsampling): Sequential convolution and max-pooling layers that reduce spatial dimensions while increasing feature depth.

Bottleneck: The deepest layer of the network where the most abstract feature representations are extracted.

Decoder (Upsampling): Transposed convolutions that restore spatial resolution.

Skip Connections: Direct links between the encoder and decoder that preserve high-resolution spatial information, which is often lost during downsampling.

Output Layer: A final 1x1 convolution with a sigmoid activation function to produce the probability map.

Dataset Organization
To ensure the model trains effectively, the data must be structured so that each input image corresponds directly to a ground-truth mask. The recommended directory structure is as follows:

Directory	Content Description
/images	Original input files (e.g., RGB photographs)
/masks	Binary ground-truth images representing the target segments
/train	Subset of data used for model weight optimization
/val	Independent data used to monitor generalization during training
