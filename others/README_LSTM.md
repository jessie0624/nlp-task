使用 pack_padded_sequence: https://blog.csdn.net/u012436149/article/details/79749409

在pytorch的RNN中，双向LSTM/GRU/RNN 必须使用pack_padded_sequence 否则，pytorch无法获取序列的长度，这样无法正确的计算双向RNN的结果。
在使用pack_padded_sequence时有个问题，即输入mini-batch序列的长度必须是从长到短排序好的，当mini-batch中的样本顺序非常重要的时候，这就有点棘手了。

pack_padded_sequence是将句子按照batch 优先原则记录每个句子的词，变化为不定长tensor,方便计算损失函数。
pad_packed_sequence是将pack_padded_sequence生成的结构转换为原先的结构，定长的tensor.


