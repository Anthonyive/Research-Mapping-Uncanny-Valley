import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SentenceTransformerModel(nn.Module):
    def __init__(self):
        super(SentenceTransformerModel, self).__init__()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # eg. tensor([[0, 1, 2]]) -> tensor([[0],[1],[2]])
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        print(position.shape, div_term.shape)
        print(pe.shape)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        print(x)
        return self.dropout(x)

PositionalEncoding(768, dropout=0.1)