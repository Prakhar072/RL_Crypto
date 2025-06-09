#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from models.FFC import FFC


# In[2]:


class FFCStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout=0.1):
        super(FFCStack, self).__init__()
        # Chain of three FFC modules
        self.ffc1 = FFC(input_dim, hidden_dim, heads, dropout)
        self.ffc2 = FFC(hidden_dim, hidden_dim, heads, dropout)
        self.ffc3 = FFC(hidden_dim, hidden_dim, heads, dropout)

        # Final convolutional layer
        self.final_conv = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        #"""
        #x shape: (batch, seq_len, input_dim)
        #"""
        x = self.ffc1(x)  # -> (batch, hidden_dim, seq_len)
        x = self.ffc2(x)  # -> (batch, hidden_dim, seq_len)
        x = self.ffc3(x)  # -> (batch, hidden_dim, seq_len)

        # Final convolution expects (batch, hidden_dim, seq_len)
        x = self.final_conv(x)  # -> (batch, hidden_dim, seq_len)

        return x  # (batch, hidden_dim, seq_len)


# In[ ]:




