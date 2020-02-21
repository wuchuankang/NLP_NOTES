# collections 包
collections 是Python的一个加强版的数据结构的一个包，在Python中内置的数据结构有str,int,list,tuple,dict，在collections中定义了deque(双向队列),OrderedDict(有序词典),defaultdict(默认初始化的字典子类)等高级的数据结构。
- OrderedDict：有序字典，按照加入到词典中的顺序进行排列，先加入的排在前面，要注意的是，即使OrderedDict 元素相同，加入元素的顺序不同，也认为是两个不同的字典。  
OrderedDict 类型经常用在nn.Module中权重参数上，因为每一层的权重有先后顺序，通过有序词典，第0层的权重先加入，自然放到第一位，这样管理各层的权重更方便。
- defaultdict：经常用来初始化指定元素数据结构的字典：
```python
from collections import defaultdict

a = defaultdict(int)
a['yes']  # 会默认是a['yes']=0
b = defaultdict(list)
b['time']  # 默认是b['time']=[]
b['first'].append(1)
print(b)  # difautdict(<class 'list'>,{'time':[], 'first':[1]})
```
