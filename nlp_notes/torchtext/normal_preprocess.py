"""
train.txt的构成是，每一行是一个文本+ 标签， 用tab分割
我们用训练集来建立词表(vocab)，这是因为在pytorch 中，如果不使用预先训练好的词向量，
那么nn.embedding 模块中会用一维标准正态分布来随机初始化词向量，在训练过程中，词向量也是学习参数；
如果使用训练集和验证集来构建词表，出自验证集中的词对应的词向量就没法更新，因为验证集不参与学习；
即使有预先使用的词向量，有时候，我们也将词向量作为学习参数；
"""
import jieba
import os
import pickle
import numpy as np

#全局变量要大写
MAX_VOCAB_SIZE = 10000
UNK, PAD = '<unk>', '<pad>'   # 前者是oov(out of vocab)，词表之外的，比如出现在验证集和测试集中，还要去除掉的词频< min_freq 的词；后者是因为我们要将每一个样本搞成长度一致的数据，这样才好放到神经网络中去训练，我们会指定数据长度，超过了就截取，不足就用<pad>填充，值得注意的是，对于RNN一类的循环神经网络，可以只满足每一批数据长度相同，不同批的数据可以不同，这是因为每一批数据长度的不同，不会影响到各层权重的维度，只是最后序列输出个数不同（此处说的是y1,...,yT），但是一般最好处理成为个批序列都相同。

def tokenizer(text):
    return [_ for _ in jieba.cut(text)]


def build_vocab(path, tokenizer, max_vocab_size, min_freq):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:

        for line in f.readlines():
            line = line.strip()  # 除去空格
            if not line:   #除去空格后，如果是空行，那么就继续
                continue
            else:
                line = line.split('\t')[0]   # 一定先尝试好，每行文本和标签是靠什么分割的，这里是用tab键分割的
                line = tokenizer(line)

                for word in line:   # 收集每个词的词频，用于将词频< min_freq的词给去掉
                    vocab[word] = vocab.get(word, 0) + 1

        word_list = sorted([_ for _ in vocab.items() if _[1]>min_freq], key=lambda x: x[1], reverse=True)   #去掉小于min_freq的词，并将词按照词频从高到低排列，用于后面词汇量> max_vocab_size 时，进行截取
        if len(word_list) > max_vocab_size:     # 词汇表进行截取
            word_list = word_list[:max_vocab_size]
        for id, item in enumerate(word_list):  #建立词汇表,  该命令等价于 vocab = {item=[0]:id for id, item in enumerate(vocab.items())} 
            vocab[item[0]] = id 
        vocab.update({UNK:len(vocab), PAD:len(vocab)+1})  #添加这两个特殊标记的索引

    return vocab

class Config():   # 类大写字母开头，函数小写字母开头
    def __init__(self, dataset_path):
        self.vocab_path = './data/vocab.pkl'
        self.embeddings_path ='./data/embeddings.npy'
        self.dataset_path = dataset_path
        self.train_path = './data/train.txt'
        self.fixed_seq_size = 40
        self.tokenizer = tokenizer
        self.max_vocab_size = 10000
        self.min_freq = 5
        self.vocab = {}
        self.vector_dim = 300


def build_dataset(config):
    if os.path.exists(config.vocab_path):
        config.vocab = pickle.load(open(config.vocab_path, 'rb'))
    else:
        config.vocab = build_vocab(config.train_path, config.tokenizer, config.max_vocab_size, config.min_freq)
        f = open(config.vocab_path, 'wb')
        pickle.dump(config.vocab, f)
        f.close()

    with open(config.dataset_path, 'r', encoding='utf-8') as f:
        content = []
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue

            line, label = line.split('\t')   # 这一步非常的值得学习，没必要分成两步写，train.txt中label已经数值化了，否则对label也要处理
            line = config.tokenizer(line)
            if len(line) < config.fixed_seq_size:
                line.extend([PAD]*(config.fixed_seq_size - len(line)))
            else:
                line = line[:config.fixed_seq_size]

            line_idx = []
            for word in line:
                idx = config.vocab.get(word, config.vocab[UNK])
                line_idx.append(idx)
            content.append((line_idx, label))
        return content

# 得到与vocab对应的词向量，这个词向量是后面作为torch.nn.Embedding.from_pretrained(Embedding)中参数Embedding，查看源码知道，这个 Embedding 参数是一个2dim的tensor，维度为(num_embeddings, embedding_dim)，这个Embedding参数会加入到module.parameters()中。所以这里我们要得到这样的一个tensor,vocab中词的索引和embedding中的索引要对应一致。
def get_embedding(pretrained_vector_path, vocab, vector_dim):
    with open(pretrained_vector_path, 'r', encoding='utf-8') as f:
        embeddings = np.random.rand(len(vocab), vector_dim)
        for i, line in enumerate(f.readlines()):
            if i == 0:   # 通过linux 的 head ./data/sgns.sogou.char 命令查看知道，第一行是词向量总数和词向量维度，所以跳过
                continue
            line = line.strip().split(' ')
            if line[0] in vocab:
                idx = vocab[line[0]]
                emb = [float(x) for x in line[1:]]
                embeddings[idx] = np.array(emb)
        return embeddings


if __name__=='__main__':
    pretrained_vectors_path = './data/sgns.sogou.char'
    config = Config('./data/train.txt')
    train_word2id = build_dataset(config)
    if os.path.exists(config.embeddings_path):
        embeddings = np.load(config.embeddings_path)
    else:
        embeddings = get_embedding(pretrained_vectors_path, config.vocab, config.vector_dim)
        np.save(config.embeddings_path, embeddings)

    print(embeddings[0])




