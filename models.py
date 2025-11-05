import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class FCNet(nn.Module):

    def __init__(self, activation_function_name):
        super(FCNet, self).__init__()
        if activation_function_name == "relu":
            self.activation_function = torch.relu
        if activation_function_name == "sigmoid":
            self.activation_function = torch.sigmoid
        # initializing fcn layers
        self.linear1 = nn.Linear(3072, 500)
        self.linear2 = nn.Linear(500, 100)
        # 10 classes in CIFAR-10
        self.linear3 = nn.Linear(100, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        # forward pass (self.activation_function)
        
        # first hidden layer + activation
        x = self.activation_function(self.linear1(x))
        # second hidden layer + activation
        x = self.activation_function(self.linear2(x))
        # classification layer (no activation here; CrossEntropyLoss in run.py expects logits)
        x = self.linear3(x)

        return x

class ConvNet(nn.Module):

    def __init__(self, activation_function_name):
        super(ConvNet, self).__init__()
        if activation_function_name == "relu":
            self.activation_function = torch.relu
        if activation_function_name == "sigmoid":
            self.activation_function = torch.sigmoid
        # layers for the convolutional neural network
        # conv1: in_channels=3 (RGB), out_channels=32, kernel_size=3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        # conv2: in_channels=32, out_channels=64, kernel_size=3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # max pool: 2x2
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # flatten
        self.flatten = nn.Flatten()
        # they told us the flattened size is 12544, and we need 10 classes
        self.linear1 = nn.Linear(12544, 10)

    def forward(self, x):
        # forward pass (self.activation_function)
        # conv → activation
        x = self.activation_function(self.conv1(x))
        # conv → activation
        x = self.activation_function(self.conv2(x))
        # pool
        x = self.maxpool2d(x)
        # flatten
        x = self.flatten(x)
        # final linear (logits)
        x = self.linear1(x)
        
        return x