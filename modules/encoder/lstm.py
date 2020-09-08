r"""
封装LSTM模块，可在forward时序传入序列长度，自动对padding做合适的处理
"""

__all__ = [
    "LSTM"
]

import torch
import torch.nn as nn 
import torch.nn.utils.rnn as rnn 

class LSTM(nn.Module):
    r"""
    LSTM模块，在提供seq_len的情况下，将自动使用pack_padded_sequence,
    同时默认将forget gate的bias初始化为1，且可以应对DataParallel中的LSTM的使用问题
    """
    def __init__(self, input_size, hidden_size=100, num_layers=1,
            dropout=0.0, batch_first=True, bidirectional=False, bias=True):
        r"""
        :param input_size: 输入x的特征维度
        :param hidden_size: 隐状态 h 的特征维度，如果bidirectional为True, 则输出维度是hidden_size*2
        :param num_layers: rnn的层数
        :param bidirectional: 若为True,使用双向RNN，反之单向
        :param batch_first: 若为True,输入和输出tesor的形状为
                    :(batch_size, seq, feature)
        :param bias:如果为false 模型不使用bias
        """
        super(LSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout, bidirectional=batch_first)
        self.init_param()
    
    def init_param(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                start, end = n//4, n//2 #TODO:check why
                param.data[start:end].fill_(1)
            else:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x, seq_len=None, h0=None, c0=None):
        r"""
        :param x: [batch_size, seq_len, input_size]输入序列
        :param seq_len: [batch_size, ] 序列长度 若为“None"，所有输入看做一样长，
        :param h0: [batch_size, hidden_size] 初始隐状态，若为 None，设为全0向量
        :param c0: [batch_size, hidden_size] 初始cell状态，若为None, 设为全0向量
        :return (output, (ht, ct)): 输出序列 和 ht,ct
                :output:[batch, seq_len, hidden_size*num_direction]
                :ht,ct:[num_layers*num_direction, batch, hidden_size]
        """
        batch_size, max_len, _ = x.size() 
        if h0 is not None and c0 is not None:
            hx = (h0, c0)
        else:
            hx = None
        if seq_len is not None and not isinstance(x, rnn.PackedSequence):
            sort_lens, sort_idx = torch.sort(seq_len, dim=0, descending=True)
            if self.batch_first:
                x = x[sort_idx]
            else:
                x = x[:, sort_idx]
            x = rnn.pack_padded_sequence(x, sort_lens, batch_first=self.batch_first)
            output, hx = self.lstm(x, hx)
            output, _ = rnn.pad_packed_sequenc(output, batch_first=self.batch_first, total_len=max_len)
            _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)
            if self.batch_first:
                output = output[unsort_idx]
            else:
                output = output[:, unsort_idx]
            hx = hx[0][:, unsort_idx], hx[1][:, unsort_idx]
        else:
            output, hx = self.lstm(x, hx)
        return output, hx 
        