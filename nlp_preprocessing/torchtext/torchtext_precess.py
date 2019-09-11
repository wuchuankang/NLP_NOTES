from torchtext import data
from torchtext.vocab import Vectors   # 用于加载预训练的词向量
from torchtext.data import TabularDataset, BucketIterator
from torch.nn import Embedding
import jieba
import os

def tokenizer(text):
    return [_ for _ in jieba.cut(text)]


# 建立处理csv文件的字段，这里batch_first=true是将batch作为第一维度，注意 TEXT这样，那么 LABEL 也应该一样
# 注意 LABEL 中的use_vocab=false，那是因为我们已经将文件的label预处理为数值的类别类型，否则也一样需要建立vocab
TEXT = data.Field(sequential=True, use_vocab=True, tokenize=tokenizer, fix_length=30, batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)

# 将用字段所定义的属性处理后读入，当csv文件有header时(就是第一行是标题，比如本文的train.csv，第一行就是标题：text, label)，要用skip——header=true来跳过。
# TEXT 中的fix_length 在这里不会进行处理，它是在生成批数据迭代器的时(BucketIterator时)进行处理的。
fields = [('text', TEXT), ('label', LABEL)]
train_data = TabularDataset(path='./data/train.csv',  format='csv', skip_header=True, fields=fields)

print(vars(train_data[0]))

# 如果使用预先好的向量，那么建立词表的时候，就要将pretrain_vectors 加载进来，这样才能够在词表和词向量之间建立联系，这里pretrain_vectors 的形式是：每一行元素用空格分开，第一个元素是词，其他元素是数值向量。pretrain_vectors 一般是保存为txt格式，此处使用的词向量就是txt格式的 ./data/sgns.sogou.char .
# 这里的pretrain——vectors加载是需要注意的：
#以这个TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))为例，默认情况下，会自动下载对应的预训练词向量文件到当前文件夹下的.vector_cache目录下，.vector_cache为默认的词向量文件和缓存文件的目录。
#即我们每做一个nlp任务时，建立词表时都需要在对应的.vector_cache文件夹中下载预训练词向量文件，如何解决这一问题？我们可以使用torchtext.vocab.Vectors中的name和cachae参数指定预训练的词向量文件和缓存文件的所在目录。因此我们也可以使用自己用word2vec等工具训练出的词向量文件，只需将词向量文件放在name指定的目录中即可。

# ./data/sgns.sogou.char 是预先下载好的词向量，
if not os.path.exists('.vector_cache'):
    os.mkdir('.vector_cache')

pretrained_vectors = Vectors(name='./data/sgns.sogou.char')
TEXT.build_vocab(train_data, vectors=pretrained_vectors, max_size=10000, min_freq=5)
#注意上面是对TEXT 字段的的属性进行build_vocab，对应的就是train_data 中的text中的内容，如果需要对label进行build_vocab，同样 : LABEL.build_vocab(train_data)，如果train.csv 文件中标题有多个标签属性，则会建立？可以参照https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/的例子自己进行尝试。
#上面建立的vocab 存储在 TEXT.vocab 中， 

# 生成批向量数据迭代器， BucketIterator 会根据 Field 中 的 fix_length 来进行padding 或者 截取，并且生成的train_iter 是原来train_data对应的词表中的索引，可参照：https://blog.csdn.net/Real_Brilliant/article/details/83117801

# 生成的结果是一个可迭代对象，用iter(train_iter))生成迭代器，用next(iter(train_iter)))查看文件
train_iter = BucketIterator(dataset=train_data, batch_size=128, shuffle=True)

# 对于验证集和测试集也一样处理生成可迭代对象，注意的是这两个集合不需要shuffle(shuffle=False)，因为shuffle是用于学习的时候缓解过拟合的一种手段。

#现在还有一个问题就是：用pytorch 框架，我们如何将词向量赋给torch.nn.Embedding？ pytorch 已经预留好了接口，要注意的是，我们传的不是 ./data/sgns.sogou.char 整个词向量，而是对应词表中的词向量，这个存储在 TEXT.vocab.vectors中


# 查看源码可知，from_pretrained 调用了本类Embedding自身，所以它实现了下面2句的功能：
#self.embed = nn.Embedding(vocab_size, embedding_dim)
# self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
Embedding.from_pretrained(TEXT.vocab.vectors)
batch = next(iter(train_iter))

# batch 中有两个属性，分别是最开始的train_data中的text 和 label,所以在定义train函数，要注意：
# for idx, batch in enumerate(train_iter):
    # input = batch.text
#     label = batch.label

print(Embedding(batch.text[0]))
