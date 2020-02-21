import re
import multiprocessing
import jieba
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def cut(string):
    return ' '.join(jieba.cut(string))

def parse_news(read_file_path, save_file_path):
    file = pd.read_csv(read_file_path, encoding='gb18030')
    file = file.fillna('')
    #将其转换成为列表处理更方便，这样一篇文档就是一个content元素
    content = file['content'].tolist()
    # 用正则表达式将标点符号去除
    pattern = re.compile(r'\w+')

    content = [pattern.findall(line) for line in content]
    content = [' '.join(line) for line in content]

    content_fenci = [cut(line) for line in content]
    with open(save_file_path, 'w') as f:
        for line in content_fenci:
            f.write(line+'\n')
           
def train_word_vector(save_file_path):
    model = Word2Vec(LineSentence(save_file_path), size=5, workers=multiprocessing.cpu_count())
    model.save('./news_model')


if __name__=='__main__':
    read_file_path = './sqlResult_1558435.csv'
    save_file_path = './result_news'
    # parse_news(read_file_path, save_file_path)
    train_word_vector(save_file_path)
