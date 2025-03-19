#!/usr/bin/env python
# coding: utf-8

# # FFC

# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import the models
from models.FE import TemporalConvNet
from models.CE import SelfAttention


# In[5]:


class FFC(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout=0.1):
        super(FFC, self).__init__()
        self.tcn1 = TemporalConvNet(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2 = TemporalConvNet(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.san = SelfAttention(embed_size=hidden_dim, heads=heads, dropout=dropout)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        x_residual = self.residual(x)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.san(x)
        return x + x_residual  # Residual connection for stability


# In[ ]:




