# CIFAR-10 Neural Network Classifier

This project implements and trains two neural network architectures, both a Fully Connected Network (FCN) and a Convolutional Neural Network (CNN), in order to classify images from the CIFAR-10 dataset using PyTorch.

## Overview
The project explores core deep learning concepts including:
- Designing neural network architectures in PyTorch  
- Training and evaluation loops  
- Activation functions (ReLU and Sigmoid)  
- Hyperparameter tuning (learning rate and activation function)

Main use is for learning rate comparison and visualization, as shown in the results.

## Models Implemented

### Fully Connected Network (FCNet)
- 3 linear layers: 3072 → 500 → 100 → 10
- Activation: ReLU or Sigmoid between hidden layers

### Convolutional Neural Network (ConvNet)
- Conv1: 3×3 kernel, 32 output channels
- Conv2: 3×3 kernel, 64 output channels
- MaxPool2d (2×2), Flatten, Linear(12544 → 10)


## Hyperparameter Search
A grid search was conducted on the CNN using combinations of learning rate and activation function.

| Learning Rate | Sigmoid | ReLU |
|---------------|----------|------|
| 1e-7 | 10.0% | 12.6% |
| 1e-3 | 37.7% | 65.1% |
| 1 | 10.0% | 10.0% |

Best configuration: Learning Rate = 1e-3, Activation = ReLU.

## Files
- models.py – Defines FCNet and ConvNet architectures.  
- run.py – Training, testing, and grid search logic.  
- data.py – Loads and normalizes the CIFAR-10 dataset.  
- requirements.txt – Dependencies for the environment.

## Usage
```
# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run training and evaluation
python run.py
```

Trained model files will be saved as:
- cifar-10-fcn.pt
- cifar-10-cnn.pt

## Results Summary
- FCN with no training achieved about 52% accuracy on CIFAR-10 after 5 epochs.
- CNN achieved about 65% accuracy and performed best with ReLU activation.  
- ReLU consistently outperformed Sigmoid due to reduced vanishing gradients.  
- Very small or large learning rates hindered training progress.

