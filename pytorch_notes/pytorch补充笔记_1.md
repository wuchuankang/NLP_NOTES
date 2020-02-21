# 补充的知识点

## tensor 基础操作
- 从接口上讲，分为两类：    
    - torch.function ，比如 torch.save
    - tensor.function ，比如 tensor.view等
对tensor的大部分操作，同时支持这两种操作，如torch.add(a,b)，a.add(b)，torch.max(a)，a.max()等等。
- 存储角度，分为两类：
    - 不修改自身操作：a.add(b),结果返回一个新的tensor
    - 修改自身： a.add_(b) ，结果仍然存储在a中
函数名以_结尾的都是inplace 方式。

## 函数参数中使用列表还是散列数的规则
当指定维度后，且还需要其他的参数，就要用列表，否则不用：
```python
a = torch.rand(3,4)  #只要指定维度，没有其他参数，不需要将维度用列表给出
a = torch.full([3,4], 1)   #指定维度，还需要给出其他参数，维度需要用列表给出
```

## 对维度的操作，dim=0,dim=1等
dim=n，就对第n-1个维度进行操作，对于一个2维矩阵，dim=0，就是对行操作，然后进行广播，其广播的意思，是有多少列，就广播多少次。没广播之前，可看成是对第一类的行操作；dim=1，想法是一样的，对行操作，然后广播；
```python
a = torch.rand(3,4)
a.sum(dim=0)    #就是每一行累加得到的结果，首先对第一列的每一行累加，然后广播
a.cumsum(dim=0) # 也是每一行累加，但是保留中间结果 
```

## .contiguous 方法
一个Tensor执行转置操作后，实际上数据并没有发生任何变化，只是读取数据的顺序变化了。这是一种节约空间和运算的方法。    
不过转置之后的数据如果需要进行.view等操作的话，由于数据在逻辑上并不连续，因此需要手动调用contiguous让数据恢复连续存储的模样。
```python
b = a.t()
b.view(-1,3)   #会报错，不连续
b.contiguous()
b.view(-1,3)
```

## 微分变量与求导
- 微分变量Variable和tensor，在0.4及以后版本中，微分变量和tensor已经合并，微分变量是计算图的关键，微分变量有3个属性：data,grad,grad_fn，所以tensor也有这三个变量。虽然是合并了，但是只有指定了reuires_grad=True 的tensor才可以求导，才算微分变量，因为这样的tensor才可以求导。
- 微分变量不能实现inplace操作，因为要对其求导，做了inplace操作，就无法追踪最开始时候该变量，求导就出错；
    ```python
    a = torch.rand(3,4, requires_grad=True)
    b = torch.rand(3,4)
    a.add_(b)   # 报错，Varible不能实现本地操作
    ```
- 微分变量从新赋值，实际是新的内存上操作，就变为了中间节点，对应与计算图，那么微分变量求导后，是对中间节点求导，最初的叶子节点被覆盖，所以不存在。这在使用的过程中要尤为注意！具体看例子：
    ```python
    a = torch.rand(3,4)
    a.is_leaf   # True
    a = a*2
    a.is_leaf   # False
    c = torch.sum(a)
    c.backward()
    a.grad   # 结果是none，因为backward中间结果导数不保存，只保存叶子节点的，其他的被清除
    ```
    **要知道的是，计算图是有向无环图，这里对叶子节点再次赋值，计算图将出现环结构，这个是忌讳！**
- 依赖与叶子节点（计算图底部的微分变量，也是我们要更新的微分变量）的中间节点的requires_grad=True是默认的。因为计算叶子节点的导数，就要计算损失函数对这中间依赖节点的导数，依赖节点必须是微分变量。但是loss.backward() 后，中间的微分变量的导数为清除，只保留叶子节点的微分变量。

- backward(gradient=None, retain_graph=None, creat_graph=None) 参数的理解
    - gradient : pytorch是不允许矩阵或者向量对矩阵或者向量直接求导的，因为最终的维度很难确定，官方文档上说，该函数的求导利用的是链式法则，而链式法则就是可以避免高纬之间的求导，而只采用标量对向量或矩阵的求导，那么高纬对高纬的求导，就可以通过维度相容原理给间接求出来，具体可参看之前写的矩阵求导。文档中说：usually the gradient of the differentiated function w.r.t. corresponding tensors 。gradient是相应的要求导的张量的微分函数的导数。具体来说就是：
    ```python
    a = torch.rand(3,4, requires_grad=True)
    b = torch.rand(3,4, requires_grad=True)
    x = a + b
    y = torch.sum(x)
    ```
    ~~现在要求$\frac {\partial x}{\partial a}$，怎么求？这是一个矩阵对矩阵的导数，结果是什么呢？先不说pytorch怎么解决的，我们应该怎么做。简单的办法就是间接求，就是求：~~
    $$
    \frac {\partial y}{\partial a} = \frac {\partial y}{\partial x}\frac {\partial x}{\partial a}
    $$
    ~~只要求出了$\frac {\partial y}{\partial a}$和$\frac {\partial y}{\partial a}$，那么另外一个就好求了。而这两个恰好是标量对矩阵的导数，是好求的，这也是用链式法则的好处。  
    pytorch 正是按照这种方式设计自动求导函数torch.autograd.backward()的。这个函数的第一个参数就是你自己定义的一个和被求导(上面的例子中的x,y)维度相同的tensor，假定为w，然后可以通过$z=torch.sum(x*w)$构造出一个标量，可以看成是损失函数，然后通过z对a的导数，间接求出来x对a的导数。~~
正确的理解是，它一般不能求出高维之间导数的，假如y是由高维x经过一些列转化后得到的，如果yy不是标量，那么y.backward(w)不是y对x的导数，它是z=torch.sum(y*x)对x的导数，也就是说pytorch的autograd是不能够实现高维之间的求导的。这样设计backward 函数的原因在于，向后传播的过程中，如果我们要求损失函数对某一层权重参数的导数，只要记录了损失函数对该层激活函数h的导数l，那么h.backward(l)就可以求出损失函数对该层权重的导数。
**这里所谓的标量，仍然是一个tensor，只是说该tensor的shape是torch.Size([])，就是0维的！**

    - retain_graph : 将计算图保存下来，这样可以继续使用图求导，要有这个参数的原因是，backward一次后，计算图就不在保留了，那么想要再使用图求导，就报错。
    ```python
    from torch.nn import functional as F
    a = torch.rand(4, requires_grad)
    p = F.softmax(4, dim=0)
    p.autograd.grad(p[0], [a], retain_graph=True)
    p.autograd.grad(p[1], [a])   # 如果没有上面计算图的保存，由于计算图清除，再次计算就会报错
    ```
    那么为什么神经网络的时候，没有将 retain_graph=True，那是因为有for循环，每个batch都会生成一个计算图，且每个batch梯度只要计算一次，所以不用保留。

    - create_graph：对反向传播过程再次构建计算图，可通过backward of backward实现求高阶导数。

- torch.autograd.backward() 和 variable.backward()
    后者在tensor类的backward()方法实现的时候，调用了前者，所以torch.autograd.backward()才是本体。
