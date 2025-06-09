#!/usr/bin/env python
# coding: utf-8

# # FFC

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import the models
from models.FE import TemporalConvNet
from models.CE import SelfAttention


# In[4]:


class FFC(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout=0.1):
        super(FFC, self).__init__()
        # TCN layers
        self.tcn1 = TemporalConvNet(input_dim, [hidden_dim], kernel_size=3, dropout=dropout)
        self.tcn2 = TemporalConvNet(hidden_dim, [hidden_dim], kernel_size=3, dropout=dropout)

        # Self-Attention
        self.san = SelfAttention(embed_size=hidden_dim, heads=heads, dropout=dropout)

        # Residual convolution
        self.ffc_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"FFC Input Shape: {x.shape}")  
        #"""
        #x shape: (batch, seq_len, input_dim)
        #1) TCN1 expects (batch, in_channels, seq_len)
        #2) TCN2 expects (batch, hidden_dim, seq_len)
        #3) SAN expects (batch, seq_len, hidden_dim)
        #"""
        # 1) Prepare x for TCN1
        # x_tcn = x.permute(0, 2, 1)  # -> (batch, input_dim, seq_len)
        # print(f"After Permute Shape (Before TCN1): {x_tcn.shape}")  
        x_tcn1 = self.tcn1(x)   # -> (batch, hidden_dim, seq_len)

        # 2) TCN2
        x_tcn2 = self.tcn2(x_tcn1)  # -> (batch, hidden_dim, seq_len)

        # 3) Prepare x_tcn2 for SAN
        x_san_in = x_tcn2.permute(0, 2, 1)  # -> (batch, seq_len, hidden_dim)
        output_san = self.san(x_san_in)     # -> (batch, seq_len, hidden_dim)

        # 4) output_tcn2 = x_tcn2 (we'll rename for clarity)
        output_tcn2 = x_tcn2  # shape (batch, hidden_dim, seq_len)

        # 5) Convolution( output_ffc(i-1) ) = convolution of the *original input x*
        #    Because x is shape (batch, seq_len, input_dim), we must permute & conv
        x_residual = x             # -> (batch, input_dim, seq_len)
        conv_res = self.ffc_conv(x_residual)         # -> (batch, hidden_dim, seq_len)

        # 6) Combine them: output_san + output_tcn2 + conv_res
        #    But note that output_san is (batch, seq_len, hidden_dim),
        #    output_tcn2 and conv_res are (batch, hidden_dim, seq_len).
        #    So we permute output_san to match (batch, hidden_dim, seq_len).
        output_san_perm = output_san.permute(0, 2, 1)  # -> (batch, hidden_dim, seq_len)

        out_sum = output_san_perm + output_tcn2 + conv_res

        # 7) Final ReLU
        out = self.relu(out_sum)

        # Return shape consistent with TCN output for chaining
        # If you want (batch, seq_len, hidden_dim) instead, you can permute again.
        return out  # shape: (batch, hidden_dim, seq_len)


# In[ ]:




