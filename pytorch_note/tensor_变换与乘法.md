# pytorch 中改变 tensor 的操作的笔记

##  转置
- permute()

    ```python
    import torch as t
    from torchtext import data 
    a = t.randn(2,3,4)
    print(a)
    b = a.permute(1,0,2)
    print(a.transpose(0,1))
    c = a.transpose(0,1)
    print(a == c)
    ```
    ```python
    tensor([[[ 0.7181, -0.3599, -1.7479, -0.9668],
         [ 0.3987, -0.9559,  0.4259, -0.0208],
         [-1.4964,  1.6284,  0.0481,  0.2063]],

        [[-0.3661,  0.2528,  0.5502, -1.7503],
         [ 1.7910, -1.8392, -1.7486, -0.7850],
         [-1.2708,  0.0360, -1.1641,  0.7730]]])
    tensor([[[ 0.7181, -0.3599, -1.7479, -0.9668],
         [-0.3661,  0.2528,  0.5502, -1.7503]],

        [[ 0.3987, -0.9559,  0.4259, -0.0208],
         [ 1.7910, -1.8392, -1.7486, -0.7850]],

        [[-1.4964,  1.6284,  0.0481,  0.2063],
         [-1.2708,  0.0360, -1.1641,  0.7730]]])
    tensor([[[True, True, True, True],
         [True, True, True, True]],

        [[True, True, True, True],
         [True, True, True, True]],

        [[True, True, True, True],
         [True, True, True, True]]])
    ```
- transpose()  
    transpose(0,1)： 意思是对第一维和第二维进行转置，容易仿照permute写成transpose(1,0)，那么就错了。

- 注：
    - 上面两者功能近似，transpose 一次只能转置两个维度，permute可以多个，但及时转置两个， permute 也得把各个维度写清楚， transpose 没有必要
    - 两者转置后，并不改变原来 tensor 中元素在内存中的位置，只是标签顺序改变了。这之后进行矩阵运算操作(加、乘)是没有问题的，但是要进行 view 操作，就必须先 顺序化，这是因为 view 是在连续的内存上操作的，它会认为 转置 之后的tensor标签顺序和其存储内存是一致的，但是实际不一致。.contiguous() 就会新生成一个 tensor, 使得标签顺序和内存顺序一致。
    - reshape 是 .contiguous().view()  等价的，这个可以实现简化步骤

## reset shape

- reshape

- view  
    ![pic](./IMG_20191009_104442.jpg)


## tensor 乘法

tensor 的乘法包括逐元素相乘、矩阵相乘，这两者都存在着广播机制。  
逐元素乘法较为简单，二维的矩阵乘法也简单，主要是高维矩阵的乘法，它的乘法规则形成的原因。  
对于神经网络来说，神经网络图中输入层的每一个节点对应一个样本的一个特征，而在实际训练中，是批处理的，批里面的每一个样本乘以的是同一个权重 W , 例如 x = [batch, d1]， W : [d1, d2]，这里可以直接用矩阵乘法就可以： torch.matmul(x,w)， 但当是自然语言处理的时候，x : [batch, seq, hidden], W : [hidden, d]， 我们的目的仍然是接一个全连接，这个可以认为是在rnn输出的隐状态加个linear全连接， 要对序列中每一个隐状态使用相同的权重，那么就可以写成 ：torch.matmul(x, W)，这里会把 W 广播为 [batch, hidden, d]，这里对一批中每一个样本对应一个权重，只是每个权重都是相同的。那么能不能做到对一个序列中的每一个元素给不同的权重呢？这个现在的实际中没有出现这种情况，rnn没有，cnn 也没有，所以这种情况暂时可以不考虑，那么对于每批对应一个不同的权重，这种情况在 multi-attention 中存在。  

更多的是，权重共享，这有两种含义： 对于所有样本而言是共享所有权重参数，而在cnn中，因为卷积操作，是特征共享权重。  
因为这种共享的原因，所以使得存在这种广播机制的矩阵乘法的存在，广播后的矩阵乘法，广播出的外层是对应元素操作，内存就是矩阵乘法。  
下面给出简单的例子：

```python
import torch as t
from torchtext import data 

a = t.ones(2,2,3)
print(a)

c = t.ones(1,2,3)
d = c + 1
e = t.cat((c,d), dim=0)
print(e.shape)

print(t.matmul(a, e.transpose(1,2)))

print(t.matmul(a, a.transpose(-2,-1)))

b = t.ones(2,3,2)
print(t.matmul(a, b))
```

```python

tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]])
torch.Size([2, 2, 3])
tensor([[[3., 3.],
         [3., 3.]],

        [[6., 6.],
         [6., 6.]]])
tensor([[[3., 3.],
         [3., 3.]],

        [[3., 3.],
         [3., 3.]]])
tensor([[[3., 3.],
         [3., 3.]],

        [[3., 3.],
         [3., 3.]]])

```
