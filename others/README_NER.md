# 命名实体的识别


## CRF
这个是参考allennlp写的，添加了一些注释
可以参考介绍: https://easyai.tech/ai-definition/ner/
项目：https://github.com/guillaumegenthial/sequence_tagging/blob/master/build_data.py
项目参考：https://github.com/Determined22/zh-NER-TF

下面我们从以下几方面讲解CRF
1. 生成模型 vs 判别模型
2. 有向图 vs 无向图
3. log linear model
4. Multinormal logistic regression
5. MEMM & label bias
6. linear CRF
7. Derivation of linear CRF

### 1. 生成模型 vs 判别模型
1）功能： 生成模型 可以用来做 生成任务和判别任务。 判别模型 用来做 判别任务。
2）目标函数：
    数据集 D = {(x_1, y_1),(x_2,y_2),..(x_n, y_n)}
    生成: maximize P(x,y)  最大化联合概率分布 \theta^* = argmaxP(x,y) = argmaxP(y|x)P(x) 这个要最大化两个概率，p(y|x)可用于判别，p(x)可用于生成（就是先验）
    判别: maximize P(y|x)  最大化条件概率分布   \theta^* = argmaxP(y|x)    只能用于判别
3）结论：
    生成模型： 生成数据，判别
    判别模型： 判别
    一般情况下，判别模型比生成模型效果好。
    数据少的时候 有可能是生成比判别好
    （因为数据少的时候，判别模型容易过拟合，防止过拟合的方法就是在损失函数里面添加正则，L1正则相当于先验为拉普拉斯分布，L2正则先验是高斯分布，这个就是生成模型的P(X)）

### 2. 有向图 vs 无向图



