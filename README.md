# Image Classification with PyTorch

This project is an example implementation of an image classification task using PyTorch. It includes convolutional neural network (CNN) models trained on the CIFAR-10 dataset.
It also includes fine-tuned ResNet18 and ResNet34 models and models used as feature extractor on the CIFAR-10 dataset.

## Overview

This project demonstrates how to build, train, fine-tune, and evaluate  simple CNNs for image classification using PyTorch. The trained models are capable of classifying images into one of the ten classes in the CIFAR-10 dataset.

## Features

- Implementation of 3 simple CNN model from scratch
- Transfer learning with ResNet18 and ResNet34
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

The models I have trained can be found in the **trained_models** folder.

_SimpleCNN_, _SimpleCNN_v2_ and _ImprovedCNN_ are my own models, trained from scratch on the CIFAR-10 dataset.

I also tried ResNet18, the smallest model of the ResNet family. Here I also trained from scratch and fine tuned the model and used it as a feature extractor. In these cases I used the weights trained on ImageNet as a starting point.
I also tried the ResNet34 model as a feature extractor.

The results are shown in the table below:

| Model        |        Mode         | Epoch | Val accuracy(%) | Augmentation |
|--------------|:-------------------:|:-----:|:---------------:|:------------:|
| SimpleCNN    |       default       |  10   |      60.45      |      No      |
| SimpleCNN_v2 |       default       |  10   |      62.58      |      No      |
| ImprovedCNN  |       default       |  10   |      77.26      |      No      |
| ResNet18     |       default       |  10   |      63.83      |      No      |
| ResNet18     |     fine tuned      |  10   |      80.80      |      No      |
| ResNet18     |  feature_extractor  |  10   |      77.27      |      No      |
| ResNet18     |  feature_extractor  |  15   |      76.58      |      No      |
| ResNet34     |  feature_extractor  |  10   |      78.01      |      No      |

