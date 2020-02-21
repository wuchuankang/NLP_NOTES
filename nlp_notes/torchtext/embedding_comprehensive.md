# pytorch Embedding 

该层是用来对读入的批样本查询向量表，从而给出每一个样本的词向量，每一个样本的词向量都是由一维高斯分布$\mathcal{N}(0,1)$来初始化的，这些词向量是可学习参数，词向量存储在对应的weight中：
```python
from torch.nn import Embedding
import torch as t

emb = Embedding(10, 3)

input = t.LongTensor([[1,2,4,5], [4,3,2,9]]) 
t.manual_seed(1)
# print(emb(input))
print(emb.weight)
print(emb.weight.shape)
embeds = Embedding(2, 5) # 2 个单词，维度 5
# 得到词嵌入矩阵,开始是随机初始化的
t.manual_seed(1)
print(embeds.weight)
print(emb.weight.requires_grad)

结果：
Parameter containing:
tensor([[-0.3061,  0.8570,  1.3456],
        [-1.1706, -1.4793, -0.5324],
        [ 0.5943, -1.2830, -0.3726],
        [ 0.3634,  1.3628, -0.3632],
        [-0.8093, -0.2383,  0.1840],
        [-1.1123, -0.2675,  0.1163],
        [-0.9258,  0.6873,  0.5257],
        [-0.7178, -0.5548, -0.4274],
        [ 1.9328, -0.7145, -0.9958],
        [-0.5129, -0.4641,  0.8564]], requires_grad=True)
torch.Size([10, 3])
Parameter containing:
tensor([[ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519],
        [-0.1661, -1.5228,  0.3817, -1.0276, -0.5631]], requires_grad=True)
True
```

当然也可以使用后训练好的词向量模型，glove或者word2vec，在模型训练过程中，词向量也可以学习，也可以不用学习（这两种情况都有，可以自己根据训练结果进行尝试）。当不用学习的时候，要进行一下处理：
```python
model.embedding.weight.requires_grad = False
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
```
要注意的是，model是自己建立module的对象，而embedding是对象的属性，例如：
```python
class embNet(nn.Module):
    def __init__(self, emb_size, vec_dim, hidden_size):
        super().__init()
        self.embedding = nn.Embedding(emb_size, vec_dim)
```
现在有个问题，那就是如何加载预训练的词向量模型？这要考虑到如何对批输入进行查表的找词向量的。  

首先输入是torch.LongTensor 型，所以对于批输入，预先要转换成整型的tensor，这是通过查询词典来实现的，词典中有每一个词的索引，所以输入就是的每一行，就是对应文本中分好的词在词典中的索引构成的，而词向量模型正好是每个词的索引为键，值是该索引对应词的词向量构成，这样一一映射，就可以查询了。  
这就意味着我们建立词典的时候，词典中某词的索引要和词向量模型中该词的词向量对应的索引要一致。这该怎么做到呢？  
