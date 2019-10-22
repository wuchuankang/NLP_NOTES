##  tensor 构造 和 特殊使用

pytorch 中经常要使用各种类型的 tensor ,各种 tensor 也经常用于做测试使用，所以对于基本的 tensor 的构造要清清楚楚。

- 随机种子的使用  
随机种子可以让每次生成的随机向量相同，这样方便调试，尤其在神经网络中。  
```python
seed = 2018
torch.manual_seed(seed)   # 这是当运行在cpu 上的时候生成的随机种子 ，seed 可以是任意的整数值
torch.cuda.manual_seed_all(seed)  # 这是在 gpu 上
a=torch.rand([1,5])
b=torch.rand([1,5])
```
结果是每次生成的随机数是相同的。

- 对于某些数据的保存

    比如我们将完成预处理的文件保存起来，可以有多种保存方式，这里选用序列化保存，也就是保存为pickle模式：
    ```python
    torch.save(data, 'xxx.pkl')
    # 加载
    data = torch.load('xxx.pkl')
    ```

- size 和 shape   
    两者皆可以求出tensor 的维度，一般在程序中 size用的比较多，可能是习惯使然：  
    ```python
    a = t.randn(2,3)
    print(a.size(1)) # 等价于 a.shape[1] ，等价于 a.size()[1] (这个用的少，明显繁琐)
    print(a.size())  # 等价于 a.shape
    ```
- 构造一维 tensor 使用的方法  
    可以具体的使用列表来转化，但当使用维度表示生成时候，有些时候要写成 (n,)
    ```python
    import torch as t
    t.LongTensor([1,2,3])
    t.FloatTensor([1,2,3])
    t.randn(2)
    t.tandint(5, (3,))
    ```

- tensor  VS Tensor  
    tensor 是 torch 的函数， Tensor 是 torch 的类，**从名字大写就可以看出来**，后者默认类型是 torch.FloatTensor ， 前者根据参数的内容而定，都是整数的就是 torch.LongTensor，否则和后者一样
    ```python
    import torch as t
    print(t.tensor([1,2,3].dtype))
    print(t.Tensor([1,2,3].dtype))
    ```
    结果：
    ```python
    torch.int64
    torch.float32
    ```

- rand VS randn VS randint  
    rand 是[0,1) 均匀分布； randn 是标准正态分布； randint 是在一定范围内的整数分布，类型：前两者是 torch.FloatTensor， 后者是 torch.LongTensor
    ```python
    print(t.rand(1,2), t.rand(1,2).dtype)
    print(t.randn(1,2), t.rand(1,2).dtype)
    print(t.randint(2, 5, [1,2]), t.randint(2, 5, [1,2]).dtype)
    ```
    结果是：
    ```python
    tensor([[0.5919, 0.1018]]) torch.float32
    tensor([[0.6152, 0.4986]]) torch.float32
    tensor([[3, 4]]) torch.int64
    ```

- arange VS linespace    

    arange 是整数序列，后者是以一定间隔生成的序列  
    ```python
    print(t.arange(10), t.arange(10).dtype)
    print(t.linspace(1,5,5))
    ```
    结果是：
    ```python
    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) torch.int64
    tensor([1., 2., 3., 4., 5.])
    ```

- masked_select 和 mask_fill 

    ```python
    a = t.randn(2,3)
    b = t.randint(0,2,(2,3))
    print(a)
    print(b)
    print(a.masked_fill(b==0, 1e-9))
    ```
    结果是：
    ```python
    tensor([[ 1.3873,  1.7678, -0.1299],
            [ 0.2665, -0.1628, -0.2398]])
    tensor([[1, 0, 0],
            [1, 1, 0]])
    tensor([[ 1.3873e+00,  1.0000e-09,  1.0000e-09],
            [ 2.6645e-01, -1.6283e-01,  1.0000e-09]])
    ```
    ```python
    a = t.randn(2,3)
    b = t.randint(0,2,(2,3))
    print(a)
    print(b)
    print(a.masked_select(b==0))
    ```
    结果是：
    ```python
    tensor([[-0.1394, -0.1454, -0.5380],
            [ 0.3365, -0.6991,  0.4861]])
    tensor([[1, 0, 0],
            [0, 0, 1]])
    tensor([-0.1454, -0.5380,  0.3365, -0.6991])
    ```

- scatter_ 和 fill_ 

    这两者都是 inplace 操作，还有对应的 scatter 和 fill  

    scatter_ 操作是 散布 的意思，以下面的为例子，0 表示维度，是对行进行散布。第二个参数每一行中的元素对应b中对应位置处的元素，而具体数值表示在a中存放的位置，  
    比如[0,1,2]对应的是b中[-0.9944,  1.2885, -0.3359]，而0 对应于 第一列中-0.9944 要存放的位置，也就是第一列第0行，1 对应 1.2885, 同时 1也表示在第二列中第一行；

    ```python
    a = t.zeros(4,3)
    b = t.randn(2,3)
    print('b:',b)
    print(a.scatter_(0, t.tensor([[0,1,2],[2,1,0]]), b))
    ```
    结果是：
    ```python
    b: tensor([[-0.9944,  1.2885, -0.3359],
            [ 2.1256, -0.1869, -1.7137]])
    tensor([[-0.9944,  0.0000, -1.7137],
            [ 0.0000, -0.1869,  0.0000],
            [ 2.1256,  0.0000, -0.3359],
            [ 0.0000,  0.0000,  0.0000]])
    ```
    fill_就简单多了，就是以某一个数替换原来所有的元素.  
    ```python
    b.fill_(0)
    ```

- torch.ne VS torch.eq  

当然也有 inplace 操作 ： torch.ne_ 和 torch.eq_  
前者是逐元素比较，如果相同就取 False ，否则取 True， 后者相反  
torch.ne(input, other) : 等价于 input.ne(other) ,pytorch 中很多这种两者都有的操作 ； 这里 other 可以是一个数(会使用广播机制)，也可以是 size 大小相同的矩阵  

```python
a = t.tensor([1,2,3])
a.ne(1)
a.eq(1)
```
结果是：  
```python
tensor([False,  True,  True])
tensor([ True, False, False])
```
这个在 nlp 求 batch loss 很有用，因为在处理数据集的时候，经常需要 padding ，使得batch data 的长度相同，但是求loss 的时候， padding 的东西不能加到损失当中，需要用这个来选出那些是原始序列。这个需要配合 tensor.masked_select 来使用。

