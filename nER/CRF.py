import torch
import torch.nn as nn
from typing import List, Optional
from utils import *

"""
注意pytorch 里面 
- cnn 输入shape是 BatchSize, Channel,H, W
- rnn 输入shape是 SequenceLen, BatchSize, InputSize
"""

class CRF(nn.Module):
    def __init__(self, 
                num_tags: int, 
                constraints: List[Tuple[int, int]] = None,
                include_start_end_transitions: bool = True) -> None:
        super().__init__()
        if num_tags <= 0: 
            raise ValueError(f'invalid number of tags:{num_tags}')
        
        self.num_tags = num_tags
        
        # 转移矩阵，从状态i转到状态j的 logit值
        self.transitions = nn.Parameter(troch.Tensor(num_tags, num_tags))

        # constraint_mask 是有效转移，（i,j）处1表示i可以转移到j，0表示i不能转移到j
        # 如果constraints为None，则表示状态都可以互相转移，反之 仅仅constraints为1的地方可以转移。
        # constraint_mask 是一个 (num_tags+2, num_tags+2)的矩阵，这里的2是start, end
        if constraints is None: # no constraints, allow all transitions
            constraint_mask = torch.Tensor(num_tags+2, num_tags+2).fill_(1.0)
        else:
            constraint_mask = torch.Tensor(num_tags+2, num_tags+2).fill_(0.0)
            # only enable the value in constraints
            for i,j in constraints:
                constraint_mask[i,j] = 1.0
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # need logits for transitioning from "start" state and to "end" state
        # 如果包含start 和 end的转移，则
        #      start_transitions 大小为（num_tags），里面元素表示 start 转移到各个tag的logits值
        #      end_transitions 大小为（num_tags），里面元素表示 各个tag可以转移到end的logits值.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions) # xavier初始化方法中服从正态分布，
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)
    
    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor=None) -> torch.Tensor:
        """
        computes the log likelihood
        """
        if mask is None:
            mask = torch.ones(**tags.size(),dtype=torch.bool)
        else:
            mask = mask.to(torch.bool)
        # log之前的分母，log之后的减数
        log_denominator = self._input_likelihood(inputs, mask)
        # log之前的分子，log之后的被减数
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        计算分母 归一化项，所有可能状态序列的likelihood总和
        """
        batch_size, sequence_length, num_tags = logits.size()

        # 注意RNN模型里面的shape是Seqlen,batchsize,inputsize/outputsize
        # 将batchsize 放到前面 
        mask = mask.transpose(0, 1).contiguous() # batch_size, seq_len, input_size(=num_tags)
        logits = logits.transpose(0, 1).contiguous() # batch_size, seq_len, input_size(=nums_tags)

        # 初始化alpha 大小为（batch_size, num_tags),里面的值为 start转移矩阵和第一个时间节点的logits值 
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0] 
        
        # 下面就开始对一个句子里面的每个单词i计算 从i-1转移到i的logits值。
        # 这里是一个batch多个句子多个状态转换一起计算的。
        # 矩阵形状为（batch_size, num_tags, num_tags） 表示（batch_size, current_tag, next_tag）
        for i in range(1, sequence_length):  
            # emit scores for time i(next tag)
            emit_scores = logits[i].view(batch_size, 1, num_tags) # 发射矩阵（从当前状态v转到下一个时间点的num_tags对应的各个得分）。我们在current_tag 扩展1维度。
            transition_scores = self.transitions.view(1, num_tags, num_tags) # 转移矩阵，从当前状态转移到下个状态的矩阵。在batch_size上添加1维度。
            broadcast_alpha = alpha.view(batch_size, num_tags, 1) #这个是初始化矩阵，当前的tag, 我们需要在下一个tag上扩展一维
            
            # 加起来所有score 然后沿着current_tag计算logexp
            inner = broadcast_alpha + emit_scores + transition_scores
            # mask=True的地方(有效值)沿着current_tag计算logsumexp，mask=False的地方保留之前的alpha不变
            alpha = logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (~mask[i]).view(batch_size, 1)
            
        # 需要计算start_end 转换的时候，对每个序列最后一个单词的状态 加上 转移到end的分数。
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha
        # 最后沿着num_tags 维度计算logsumexp,返回结果维度为（batch_size,）这个就是所有可能状态的加和得分，即归一化项
        return logsumexp(stops)

    def _joint_likelihood(self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        计算目标tag的得分，就是input->tag的得分
        """
        batch_size, sequence_length, _ = logits.data.shape

        # batch_size first
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # 从start tag的第一个tag开始
        if self.include_start_end_transitions:
            scores = self.start_transitions.index_select(0, tags[0])
        else:
            scores = 0.0
        
        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i+1] #（batch_size,）
            # 计算当前tag 转移到下一个tag的 转移得分
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            # 计算当前tag的所有发射得分，就是当前i为current_tag时，发射到下一个i+1时刻各个状态的得分
            emit_score = logit[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            # 计算 i时刻的current_tag到下面所有tag的得分总和。
            score = score + transition_score * mask[i+1] + emit_score * mask[i]
        # 对每个句子，我们都应该找到各个句子最后一个tag->stop。
        last_tag_index = mask.sum(0).long() - 1 # mask 为1的地方是有值，mask为0的地方没有值，mask沿着0维加起来就是各个序列的句子长度
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
        
        # 计算每个句子从最后一个状态 转移到stop状态的得分
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0
        
        last_inputs = logits[-1] # batch_size, num_tags
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1)) # batch_size,1
        last_input_score = last_input_score.squeeze() # batch_size

        score = score + last_transition_score + last_input_score * mask[-1]
        return score

    def viterbi_tags(self, logits: torch.Tensor, mask: torch.BoolTensor=None, top_k:int=None) \
                    -> Union[List[VITERBI_DECODING],List[List[VITERBI_DECODING]]]:
        """
        通过viterbi算法，获得给定序列的最可能的tags
        返回list的长度与batch_size 一样大
        返回list里面的元素是tuple（tag_sequence, viterbi_score）
        """
        if mask is None:
            mask = torch.ones(*logits.shape[:2], dytpe=torch.bool, device=logits.device)
        if top_k is None:
            top_k, flatten_output = 1, True
        else:
            flatten_output = False
        
        _, max_seq_length, num_tags = logits.size()
        logits, mask = logits.data, mask.data

        start_tag, end_tag = num_tags, num_tags+1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.0)

        # 对于 constraint 里面，可以转移的地方是1，不可以转移的是0，对于不可以转移的地方 给一个特别小的转移分数
        constrained_transitions = self.transitions * self._constraint_mask[:num_tags, :num_tags] \
                                    + (-10000.0) * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data
        
        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = self.start_transitions.detach() \
                                * self._constraint_mask[start_tag, :num_tags].data \
                                + -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = self.end_transitions.detach() \
                                * self._constraint_mask[:num_tags, end_tag].data \
                                + -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)
        for prediction, prediction_mask in zip(logits, mask): #得到的L*N的预测结果 找到一个最佳的路径。
            mask_indices = prediction_mask.nonzero().squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]

            # 其实tag有可能是任何一个
            tag_sequence.fill_(-10000.0)
            # 在0时刻，有start tag
            tag_sequence[0, start_tag] = 0.0
            tag_sequence[1:(sequence_length+1), :num_tags] = masked_prediction
            tag_sequence[sequence_length+1, end_tag] = 0.0 # 最后时刻必须是END tag（这个是logit值，END tag 概率为1, logit 1 = 0）

            viterbi_paths,viterbi_scores = viterbi_decode(
                                    tag_sequence = tag_sequence[:(sequence_length+2)],
                                    transition_matrix = transitions,
                                    top_k = top_k)
            top_k_paths = []
            for viterbi_path, viterbi_score in zip(viterbi_paths, viterbi_scores):
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)
        
        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]
        return best_paths




