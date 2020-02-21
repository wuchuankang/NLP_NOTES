# pytorch 中的 Dataset 类和 Dataloader 类
在使用pytorch，关键麻烦的问题在于数据的预处理和数据的加载，其中数据的加载在pytorch 中可以通过自定义的数据集对象来实现。 数据集对象被抽象为 Dataset 类，自定义的数据集对象要继承 Dataset 类， 并且实现两个 Python 魔法方法：

- __getitem__ : 返回一个数据或者是一个样本，obj[index] 等价于调用 obj.__getitem__(index)，**注意参数是index**
- __len__ : 返回样本的数量， len(obj) 等价于调用 obj.__len__()
    
## 关于这两个魔法的理解
通过这两个，可以创造出一个容器 container 出来(像list，tuple，dict)，这可以用在定义 pytorch 数据集类。

```python
from torch.utils.data import Dataset
import os

class pic(Dataset):
    def __init__(self, path):
        imgs = os.listdir(path)
        self.imgs = [os.path.join(path, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1

        return img_path, label  #返回这两个，才能实现构造容器

def __len__(self):
    return len(self.imgs)


if __name__=='__main__':
    pics = pic('./pic')
    print(len(pics))
    for i,j in pics:   #创造出一个容器的类
        print(i, j)
```

## Dataloader 类
Dataset 只负责数据的抽象，一次只能通过__getitem__()调用一个数据；但是训练神经网络的时候，都是对 batch 数据进行操作，同时可能对数据进行shuffle 或 并行加速加载数据，这时候就需要用 Dataloader 类了。定义如下：
```python
a =  Dataloader(dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
```
- dataset 是数据集对象，这里有一点，就是单个的数据集对象应该处理成什么样子的，这要看具体采用的是什么方法，对于神经网络来说，要求每一个样本的特征维度相同， 当处理的是时序问题，要求每个序列长度相同，序列中的每一元素的特征表示维度要相同，一个例子就是文本情感分析，要求每个文本长度要相同，这个可以通过截断和补齐进行处理成为一样的长度。

- batch_size ： 是将从dataset中读取的文件组合成batch，每个batch的大小

- collate_fn : 是组合成为批的方式，是一个函数，使用默认的即可。  

要注意的是，这是将数据一次性的加载到内存中的！对内存是有要求的，内存不够，可以尝试将数据集分批加载。  
另外对dataset，也不一定非要是Dataset对象，只要有__getitem__,__len__，不继承Dataset也是可以的。
