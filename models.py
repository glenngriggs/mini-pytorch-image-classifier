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
        # TODO: initialize the layers for the fully-connected neural network (please do not change layer names!)
        self.linear1 = nn.Linear(3072, 500)
        self.linear2 = nn.Linear(500, 100)
        # 10 classes in CIFAR-10
        self.linear3 = nn.Linear(100, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        # TODO: complete the forward pass (use self.activation_function)
        
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
        # TODO: initialize the layers for the convolutional neural network (please do not change layer names!)
        self.conv1 = None
        self.conv2 = None
        self.maxpool2d = None
        self.flatten = None
        self.linear1 = None

    def forward(self, x):
        # TODO: complete the forward pass (use self.activation_function)
        
        return x