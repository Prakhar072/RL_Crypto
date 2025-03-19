#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
from models.FFC import FFC

class FullModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, num_ffc=3, dropout=0.1):
        super(FullModel, self).__init__()
        self.ffc_layers = nn.ModuleList([FFC(input_dim if i == 0 else hidden_dim, hidden_dim, heads, dropout) for i in range(num_ffc)])
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        for ffc in self.ffc_layers:
            x = ffc(x)
        x = x.permute(0, 2, 1)  # Change to (batch, embed_size, seq_length)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # Back to (batch, seq_length, embed_size)
        return x

