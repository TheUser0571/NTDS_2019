# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 10:06:51 2019

@author: kay-1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(NeuralNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc1 = nn.Linear(self.in_size, int(1.5 * self.out_size))
        self.fc2 = nn.Linear(int(1.5 * self.out_size), int(2 * self.out_size))
        self.fc3 = nn.Linear(int(2 * self.out_size), int(1.5 * self.out_size))
        self.fc4 = nn.Linear(int(1.5 * self.out_size), self.out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
        
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,33), padding=(2,16), padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,33), padding=(2,16), padding_mode='zeros')
        self.fc1 = nn.Linear(12, 12)
        self.fc2 = nn.Linear(12, 12)


    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x.squeeze())
        x = self.conv2(x.unsqueeze(1))
        x = self.fc2(x.squeeze())
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
    

class GCN(nn.Module):
    def __init__(self, A, D):
        super(GCN, self).__init__()
        self.A = torch.cat((torch.Tensor(A), torch.ones(A.shape[0], 1)), dim=1)
        self.D = torch.Tensor(np.diag(np.diag(D)**-1))
        self.W1 = nn.Parameter(torch.rand(self.A.shape))
        self.W2 = nn.Parameter(torch.rand(self.A.shape))
        self.W3 = nn.Parameter(torch.rand(self.A.shape))
        #self.relu = nn.ReLU()
        
    def forward(self, X):
        X = X.T
        X = torch.cat((torch.Tensor(X), torch.ones(1, X.shape[1])), dim = 0) # adding bias node
        X = self.D.mm(self.A*self.W1).mm(X)
        #X = self.relu(X) # could be added - maybe even better?
        X = torch.cat((torch.Tensor(X), torch.ones(1, X.shape[1])), dim = 0) # adding bias node
        X = self.D.mm(self.A*self.W2).mm(X)
        #X = self.relu(X) # could be added - maybe even better?
        X = torch.cat((torch.Tensor(X), torch.ones(1, X.shape[1])), dim = 0) # adding bias node
        X = self.D.mm(self.A*self.W3).mm(X)
        return X.T
    def reset_parameters(self):
        self.W1 = nn.Parameter(torch.rand(self.A.shape))
        self.W2 = nn.Parameter(torch.rand(self.A.shape))
        self.W3 = nn.Parameter(torch.rand(self.A.shape))
        
        
    
        
        
        
        
        