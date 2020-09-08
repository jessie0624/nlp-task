import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from modules.CRF import *
from modules.BILSTM import *
from modules.utils import *
from typing import List, Optional

torch.manual_seed(1)
START_TAG, STOP_TAG = "<START>", "<STOP>"
def argmax(v): # v:是第1个维度
    _, idx = torch.max(vec, 1) # value index
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast))

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, 
                tag_to_ix: List[], 
                embedding_dim: int=512, 
                hidden_dim: int = 512,
                rnn_layers: int = 1,
                dropout: float = 0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = BiLSTM(embedding_dim, hidden_dim, rnn_layers, dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.crf = CRF(num_tags=self.tagset_size, include_start_end_transitions=True)

        self.hidden = self.init_hidden()

    def init_hidden(self): ## TODO: init hidden
        return (torch.randn(2, 1, self.hidden_dim//2),
                torch.randn(2, 1, self.hidden_dim//2))

    def _get_lstm_features(self, sentence, mask):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1) # batch_size * 1 * embedding_size
        lstm_out, self.hidden = self.bilstm(sentence, mask)
        lstm_feats = self.hidden2tag(lstm_out.view(len(sentence), self.hidden_dim))
        return lstm_feats
    
    def forward(self, sentence, sentence_mask):
        lstm_feats = self._get_lstm_features(sentence, sentence_mask)
        best_paths = CRF.viterbi_tags(lstm_feats)
        return best_paths
        
        