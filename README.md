# CIFAR-10-Classification-with-PyTorch
This project demonstrates building, training, and evaluating Fully Connected and Convolutional Neural Networks on the CIFAR-10 dataset using PyTorch. It also includes visualization of sample images, model architectures, and performance comparison.

## Table of Contents

- [Installation](#installation)  
- [Dataset](#dataset)  
- [Project Overview](#project-overview)  
- [Models](#models)  
- [Training & Evaluation](#training--evaluation)  
- [Results](#results)  
- [Discussion](#discussion)  
- [Usage](#usage)  
- [References](#references)  

---

## Installation

* pip install torch torchvision torchviz matplotlib scikit-learn

## Dataset

* Used the CIFAR-10 dataset, which contains: 60,000 32x32 color images in 10 classes, 50,000 training images and 10,000 test images
* The dataset is automatically downloaded using torchvision.datasets.CIFAR10.

## Project Overview

### Data Loading & Visualization

* Load CIFAR-10 using PyTorch datasets and transforms.

* Visualize example images from the training dataset.

### Model Architectures

* Fully Connected Neural Network (FCNN)

* Convolutional Neural Network (CNN)

### Model Visualization

* Use torchviz to visualize the network structures.

### Training & Evaluation

* Train both models using CrossEntropyLoss and SGD optimizer.

* Evaluate models using Accuracy and F1 Score.

### Results Comparison

*Plot Accuracy and F1 Score of both models.

### Models
#### Fully Connected Neural Network (FCNN)

* Input: Flattened 32x32x3 images

* Hidden Layer: 128 neurons with ReLU activation

* Output Layer: 10 neurons with Softmax activation

#### Convolutional Neural Network (CNN)

* Conv1: 3 → 32 filters, 3x3 kernel

* Conv2: 32 → 64 filters, 3x3 kernel

MaxPooling: 2x2

* Fully Connected: 6488 → 128 → 10

### Training & Evaluation

* Training is done for 5 epochs (adjustable)

* Device: GPU if available, else CPU

* Batch size: 64

* Metrics: Accuracy & Weighted F1 Score



Run the Jupyter Notebook or Python script

Visualize dataset images and model architectures

Train the models and compare performance metrics

