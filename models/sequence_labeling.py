r"""
本模块实现了3种序列标注的模型
"""

__all__ = [
    'SeqLabeling',
    'AdvSeqLabel',
    'BiLSTMCRF'
]

import torch
import torch.nn as nn
import torch.nn.functional as F 

from .base_model import BaseModel
from ..modules import decoder, encoder
from ..modules.decoder import ConditionalRandomField
from ..modules.encoder import LSTM
from ..modules.decoder.crf import allowed_transitions

class BiLSTMCRF(BaseModel):
    r"""
    结构为 embedding + BiLSTMCRF + FC + Droput + CRF
    """
    def __init__(self, embed, num_classes, num_layers=1, 
                hidden_size=100, dropout=0.5, target_vocab=None):
        r"""
        :param embed: embed输入 可以用各种embdding， 或者tuple 指明num_embedding， dimension
        :param num_classes: 一共多少个类
        :param num_layers: BiLSTM 的层数
        :param hidden_size: BiLSTM的hidden_size, 实际hidden size 为该值得2倍
        :param dropout: dropout的概率，0为不进行dropout
        :param target_vocab: Vocabulary对象，target与index的对应关系，
                            如果传入该值，将自动避免非法的解码序列
        """
        super().__init__()
        self.embed = get_embeddings(embed)

        if num_layers > 1:
            self.lstm = LSTM(self.embed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size,
                            bidirectional=True, batch_first=True, droput=dropout)
        else: # 如果只有一层lstm的话 就没必要加dropout，如果是多层的话可以添加dropout
            self.lstm = LSTM(self.embed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size,
                            batch_first=True)                   
        self.droput = nn.Droput(dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)

        trans = None
        if target_vocab is not None:
            assert len(target_vocab) == num_classes, "The number of classes should be same with the length of target vocabulary"
            trans = allowed_transitions(target_vocab.idx2word, include_start_end=True)
