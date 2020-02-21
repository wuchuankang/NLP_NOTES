# pandas 读取csv 文件
 - 以读取和处理一个'news.csv' 文件为例：
 ```Python
 import pandas as pd
 f = pd.read_csv('news.csv', encoding='utf-8')  # encoding根据csv文本具体编码而定，这里假设是'uft-8',编码不对，打不开文件
 print(f.colums)   #列出每列的标题
 f  = f.fillna('')
 content = f['cotent'].tolist() # f是pandas.series对象， f['content']是pandas.series.Series对象，这里假设new.csv存在一列为content的内容，且该内容使我们想要得到的东西
 ```
