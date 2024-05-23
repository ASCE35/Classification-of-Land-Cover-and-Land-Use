This folder stores the code implementation of the [*ViT*](https://arxiv.org/abs/2010.11929) model and the corresponding experimental results of land use and land cover classification, as a comparison model for the Swin Transformer algorithm used in this work. The pretrained weights we used can be downloaded from [*here*](https://github.com/google-research/vision_transformer).

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
![alt](/data/vit_train_acc.png "Figure 2. The prediction accuracy of the model on the training set")  <br/>
![alt](/data/vittrainloss.png)  <br/>
![alt](/data/ViT_val_acc.png)  <br/>
![alt](/data/vit_val_loss.png)  <br/>

Some prediction results are shown below. <br/>
![alt](/data/ViT_test1.png "Figure 6. The prediction result of the input image")  <br/>
![alt](/data/ViT_test2.png "Figure 7. The prediction result of the input image")  <br/>
![alt](/data/ViT_test3.png "Figure 8. The prediction result of the input image")  <br/>
![alt](/data/ViT_test4.png "Figure 9. The prediction result of the input image")  <br/>


Confusion matrix and  predition accuracy for the test set. <br/>
![alt](/data/ViT_Confusion_Matrix.png "Figure 10. Confusion matrix for the test set.")  <br/>
![alt](/data/vit_model_acc.png "Figure 11. Confusion matrix for the test set.")  <br/>

<br/>
The predition accuracy of the Swin Transformer for the same test set is shown below. <br/>
![alt](/data/Swin_Transformer_model_acc.png) <br/>
<br/>
From the above figures, we can see that the Swin Transformer achieves a higher prediction accuracy (98.1% vs 88.4%) and performs better in land cover and land use classification task.
