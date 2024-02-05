# Image Classification with PyTorch

This project is an example implementation of an image classification task using PyTorch. It includes 3 simple convolutional neural network (CNN) model trained on the CIFAR-10 dataset.

## Overview

This project demonstrates how to build, train, and evaluate a simple CNN for image classification using PyTorch. The trained model is capable of classifying images into one of the ten classes in the CIFAR-10 dataset.

## Features

- Implementation of 3 simple CNN model
- Training and evaluation scripts
- Basic visualization of training progress
- Save and load trained models
- Inference script for making predictions

## Getting Started

### Prerequisites

- Python 3
- PyTorch
- Matplotlib (for visualization)
- Scikit-learn

### Usage
To train the model:
```bash
python train.py
```
To evaluate the model:
```bash
python evaluate.py
```
For inference:
```bash
python inference.py
```

## Trained Models

| Model        |        Mode         |  Epoch  |  Val accuracy(%)  |
| ------------ |:-------------------:|:-------:|:-----------------:|
| SimpleCNN    |       default       |   10    |       60.45       |
| SimpleCNN_v2 |       default       |   10    |      62\. 58      |
| ImprovedCNN  |       default       |   10    |       77.26       |
| ResNet18     |       default       |   10    |       63.83       |
| ResNet18     |     fine tuned      |   10    |       80.80       |
| ResNet18     |  feature_extractor  |   10    |       77.27       |
| ResNet18     |  feature_extractor  |   15    |       76.58       |

