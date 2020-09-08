from torch.nn import functional
import numpy as np 
import torch 
from torch import nn 
from torch.nn import init
from torch.nn.utils import rnn as rnn_utils
from typing import List, Optional
import math

class BiLSTM(nn.Module):
    def __init__(self, embedding_size: int = 768, 
                hidden_dim: int = 512,
                rnn_layers: int = 1,
                dropout: float = 0.5):
        super(BiLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size,
                            hidden_dim//2,
                            rnn_layers,
                            batch_first=True,
                            bidirectional=True)
    
    def forward(self, input_, input_mask):
        length = input_mask.sum(-1)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_ = input_[sorted_idx]
        packed_input = rnn_utils.pack_padded_sequence(input_, sorted_lengths.data.tolist(), batch_first=True)
        output, (hidden_dim, _) = self.lstm(packed_input)
        padded_outputs = rnn_utils.pad_packed_sequence(output, batch_first=True)[0]
        _, reversed_idx = torch.sort(sorted_idx)
        return padded_outputs[reversed_idx], hidden[:, reversed_idx]
    
    @classmethod
    def create(cls,  *args, **kwargs):
        return cls(*args, **kwargs)


