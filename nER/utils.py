
import torch

def logsumexp(tensor: troch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim))
    
def viterbi_decode(tag_matrix: torch.Tensor,
                   transition_matrix: torch.Tensor,
                   allowed_start_transitions: torch.Tensor = None,
                   allowed_end_transitions: torch.Tensor = None，
                   top_k: int = None):
    if top_k is None:
        top_k, flatten_output = 1, True
    elif top_k >= 1:
        flatten_output = False
    else:
        raise ValueError(f'topk must be either None or >=1')
    seq_len, num_tags = list(tag_matrix.size()) # 这个应该是初始化概率矩阵
    has_start_end_restrictions = allowed_end_transitions is not None or allowed_start_transitions is not None
    if has_start_end_restrictions:
        if allowed_end_transitions is not None:
            allowed_end_transitions = torch.zeros(num_tag)
        if allowed_start_transitions is not None:
            allowed_start_transitions = torch.zeros(num_tag)
        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix
        
        allowed_end_transitions = torch.cat([allowed_end_transitions, torch.tensor([-math.inf, -math.inf])])
        allowed_start_transitions = torch.cat([allowed_start_transitions, torch.tensor([-math.inf, -math.inf])])
        # new_transition_matrix[-2]: START
        # new_transition_matrix[-1]: END # 转到其他 tag
        new_transition_matrix[-2, :] = allowed_start_transitions
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix
    
    tag_observations = [-1 for _ in range(seq_len)]
    START, END = num_tags-2, num_tags-1
    if has_start_end_restrictions:
        tag_observations = [START] + tag_observations + [END] 
        # 在每个obs对应的tag 里面 添加上START, END 就是矩阵的第一行和最后一行
        extra_tags_sentinel = torch.ones(seq_len, 2) * -math.inf
        tag_matrix = torch.cat([tag_matrix,extra_tags_sentinel], axis=-1)
        # 在序列前后添加上START, END
        zeros_sentinel = torch.zeros(1, num_tags)
        tag_matrix = torch.cat([zeros_sentinel, tag_matrix, zeros_sentinel], axis=0)
        seq_len = seq_len + 2
    
    path_scores = []
    path_indices = []
    if tag_observations[0] != -1: # START
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.0 
        # 给第一列START位置的值设为最大
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_matrix[0,:].unsqueeze(0))
    
    for timestep in range(1, seq_len):
        summed_potentials = path_scores[timestep-1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags) # num_tags * num_tags

        max_k = min(summed_potentials.size()[0], top_k) #最多返回top_k个方案
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
        # 从前一个tag v 转移到下一个tag u，选出从哪个tag v转到 特定的u是最大的
        # 所以对于每一列就是每个next_tag 找到一个最大值
        # scores 是 每一列选取的按从大到小的值 [top_k, seq_len]，paths是[top_k, seq_len]
        # scores[0,i]第i个单词的最大得分，paths[0, i]这个最大得分的来自哪个index
        observation = tag_observations[timestep]
        if observation != -1: # 说明已经有tag了
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(scores + tag_matrix[timestep,:])
        path_indices.append(paths.squeeze())
    
    # 至此已经有score和path_indices
    path_scores_v = path_scores[-1].view(-1)
    max_k = min(path_scores_v.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k,dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        viterbi_path.reverse()

        if has_start_end_restrictions:
            viterbi_path = viterbi_path[1:-1]
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)
    
    if flatten_output:
        return viterbi_paths[0], viterbi_scores[0]
    
    return viterbi_paths, viterbi_scores 


