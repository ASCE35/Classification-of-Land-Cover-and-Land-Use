This folder stores the code implementation of the [*ViT*](https://arxiv.org/abs/2010.11929) model and the corresponding experimental results of land use and land cover classification, as a comparison model for the Swin Transformer algorithm used in this work.

## Method:
![alt](/data/ViT.png "Figure 1.  The architecture of the ViT.")
<br/>
From the model architecture diagram of ViT we can see that ViT contains 3 main parts:
* **Linear Projection of Flattened Patches:** <br/>
In the ViT model, the input image is first divided into fixed-size, non-overlapping patches. Each patch is then flattened into a vector. These flattened vectors are projected into a higher-dimensional embedding space through a linear transformation, typically implemented with a fully connected layer. This step preserves the original pixel information of each patch while also capturing some relationships between patches, providing rich feature representations for subsequent processing. <br/>
* **Transformer Encoder:** <br/>
These patch embeddings, after adding positional embeddings, are fed into multiple Transformer encoder layers. Each encoder layer consists of a multi-head self-attention mechanism and a feed-forward neural network. The self-attention mechanism captures long-range dependencies between patches across the entire input sequence, while the feed-forward network applies non-linear transformations to the data. The encoder layers also use layer normalization and residual connections to stabilize the training process and enhance model performance.<br/>
* **MLP Head:** <br/>
After processing through all the Transformer encoder layers, feature representations are extracted from the final layer's output. These representations are typically pooled, either through average pooling or by selecting a specific token (e.g., a classification token), to obtain a fixed-length vector. This vector is passed through a multi-layer perceptron (MLP), which includes one or more fully connected layers with activation functions, to produce the final classification or task-specific prediction. In this work, the output layer dimension of the MLP head is modified to 10, corresponding to the land use and land cover categories. <br/>

## Results:
Below shows the graph of accuracy and loss against epochs for both training and testing data. And the model was trained for 20 epochs.
