import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from copy import deepcopy
import gymnasium as gym
# from stable_baselines3 import PPO

class Model(torch.nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden_dim=256, softmax=False):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(in_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l3 = torch.nn.Linear(hidden_dim, out_dim)
        self.softmax = softmax

    def forward(self, x):
        x = F.relu(self.l1(x.cuda()))
        x = F.relu(self.l2(x.cuda()))
        if self.softmax:
            print("Model_forward_end1")
            return F.softmax(self.l3(x))
        # print("Model_forward_end2")  
        return self.l3(x)
        

class LinearModel(torch.nn.Module):
    def __init__(self, in_dim=1, out_dim=1, bias=False):
        super(LinearModel, self).__init__()
        self.l1 = torch.nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x): 
        return self.l1(x)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.uniform_(-1.0, 1.0)