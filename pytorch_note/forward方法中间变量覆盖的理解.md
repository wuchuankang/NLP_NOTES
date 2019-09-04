# forward的时候中间变量覆盖对backward没有影响
这个问题没有完全搞清楚，看下面的一个例子：
```python
import torch  as t
import torch.nn as nn
from torch.nn import functional as F
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.lin1 = nn.Linear(3,4)
        self.lin2 = nn.Linear(4,2)
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        return self.lin2(x)
    
x = t.randn(4,3)
layer = net()
inputs = layer(x)
target = t.randint(2,(4,))
criteria = nn.CrossEntropyLoss()
loss = criteria(inputs, target)
loss.backward()
print(layer.lin1.weight.grad)
结果是：
tensor([[ 0.0278,  0.0192,  0.0481],
        [ 0.0041,  0.0039,  0.0063],
        [-0.0059, -0.0050, -0.0098],
        [ 0.0068,  0.0049,  0.0120]])
```
可见并没有因为在forward中间变量的使用x而使得loss.backward无法准确求得。这是因为虽然用了重名，但是中间变量的地址是不同的，并没有在本地操作(in-place)，每个中间变量存储的时候都会有一个version counter，用来标志变量，在反向求导的时候，依靠的是version conter 来取中间变量的，而不是靠变量alias，也就是变量名。所以中间变量名不重要，这也带来了好处就是，我们不必要为中间变量取不同的名字，尤其当网络层数很大的时候，取名将是一个很难的问题，还有一个就是当我们定义的网络，使用了nn.Sequential容器的时候，那么写forward函数将尤其简单：
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
                    nn.Conv2d(3,3,3),
                    nn.BatchNorm2d(3),
                    nn.ReLU())
        
    def forward(self,x):
        return self.layers(x)
```
