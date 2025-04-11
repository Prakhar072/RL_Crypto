#!/usr/bin/env python
# coding: utf-8

# # CE

# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        # Query, Key, Value linear layers
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

        # Final linear layer after attention
        self.conv = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape

        # Transform x into Q, K, V
        Q = self.query(x)  # (batch_size, seq_length, embed_dim)
        K = self.key(x)    # (batch_size, seq_length, embed_dim)
        V = self.value(x)  # (batch_size, seq_length, embed_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        M = torch.matmul(attention_weights, V)

        M = M.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_size)

        # Apply convolution instead of FC layer
        M = M.permute(0, 2, 1)  # Change to (batch, embed_size, seq_length) for Conv1d
        result = self.conv(M)  # Convolution operation
        result = result.permute(0, 2, 1)  # Back to (batch, seq_length, embed_size)
        result = self.dropout(result)
        return result

    def save_model(self, filepath="self_attention.pth"):
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    # Load model state
    def load_model(self, filepath="self_attention.pth"):
        self.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")


# In[ ]:




