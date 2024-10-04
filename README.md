
This repository contains an implementation of an image classification model using deep learning techniques to classify micro-organisms from the **Micro-Organism Image Classification dataset**. The project utilizes **Convolutional Neural Networks (CNNs)** to extract spatial features and improve classification accuracy on biological images.

## Introduction

This project focuses on classifying microscopic images of micro-organisms into different categories using deep learning techniques. The goal is to build a robust model that can assist in automating the identification of micro-organisms, which has potential applications in healthcare and biological research.

## Dataset

The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/mdwaquarazam/microorganism-image-classification). It contains **750 images** across **8 different classes** of micro-organisms.

### Dataset Summary:
- **Number of Images**: 750
- **Number of Classes**: 8
- **Image Size**: Varies

## Model Architecture

The model employs a **Convolutional Neural Network (CNN)** to learn the spatial features from the images. The architecture consists of multiple convolutional and pooling layers followed by fully connected layers to classify the micro-organisms.

### Key Layers:
1. **Convolutional Layers**: Extract spatial features from the input images.
2. **Pooling Layers**: Downsample the features while preserving important information.
3. **Fully Connected Layers**: Classify the images based on the learned features.

### Improvements:
- Initial accuracy: 30%
- Improved to 40% by experimenting with different CNN architectures and customizations.

### Data Augmentation:
- Rotation
- Flipping
- Zooming

These techniques were applied to enhance the robustness of the model and avoid overfitting given the limited dataset size.

## Setup & Usage

### Requirements:
- Python 3.x
- PyTorch
- OpenCV
- Numpy
- Matplotlib


