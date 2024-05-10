# Classification-of-Land-Cover-and-Land-Use
This repository contains a collection of Jupyter Notebooks and necessary Python scripts that demonstrate the application of a deep learning image classification algorithm named Swin Transformer for the classification of satellite imagery data. <br/>

**For implementation:**
* Dataset can be downloaded from the link [*here*](https://github.com/phelber/eurosat?tab=readme-ov-file).
* Run the notebook to sequentially install the required packages, configure the experimental environment, and train the model.

## The problem to be tacke:
Utilizing satellite imagery for land cover and land use classification plays a crucial role and holds significant meaning in fields such as environmental management, land planning, agricultural development, and natural resource conservation. However, the spatial distribution of different land cover types is complex, making the improvement of classification accuracy a challenging task. With the continuous development of computing power and deep learning, image classification networks based on CNN (Convolutional Neural Networks) and Transformers have been extensively researched and applied. Therefore, this work aims to employ deep learning techniques, particularly by integrating the strengths of both CNNs and Transformer networks, to achieve precise classification of land cover and land use. <br/>

## Overview:
![alt](data/overview.png "Figure 1.  Illustration of the classification of land cover and land use.")

Satellite scanning of the earth yields images that can be cropped and fed into artificial intelligence algorithms to obtain land use and land cover classification results. The dataset used in this work is the RGB version of EuroSAT, which is based on multispectral images from the Sentinel-2 satellite. The dataset contains a total of 10 categories, each containing 2000-3000 images with a resolution size of 64*64 pixels, and each pixel represents a spatial coverage of 10 meters.

The flowchart of the methodology used for land cover and land use classification adopted in this work is shown in Fig. 1. Based on the pre-trained powerful image classification algorithm Swin Transformer, the algorithm is modified and fine-tuned using the EuroSAT image dataset. The image is first partitioned into patches, then flattened and positional coding is added, and then fed into the Transformer model for multi-head self-attention learning. Finally, the land use and land cover type of the area represented by this image is output using the fully connected layers.
