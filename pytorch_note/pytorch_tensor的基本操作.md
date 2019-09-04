# torch 中tensor的基本操作
- tensor是由torch.Tensor类所定义的，在Tensor类中还定义了tensor的各种操作和属性，下面的基本操作，都是定义在tensor类中。当然还有比较重要的Tensor.backward()。
- 这里对维度要说明一下，在torch中没有单纯的数据，每个数据都有维度，即使是torch.tensor(1.0) 也有维度，维度是0。
- tensor是张量，也就是torch中所有的数据都是张量，其中0维张量对应Python中的数字，1维（也叫向量）对应Python中的list，对应numpy 中的一维array数组，2维（矩阵）对应numpy 的2维array数组。
- pytorch 使用自动微分变量来实现动态图的计算，自动微分变量原来用Variable来构造，在0.4以及后续版本中，自动微分变量和张量Tensor进行了合并，即任意一个张量都是一个自动微分变量，但是在对要微分的变量要将requires_grad参数设置为1
```python
w = torch.rand(3,4, requires_grad=True)
```
- dim参数很重要，在各种函数中都要指定维度，比如max, argmax, softmax, topk, norm, mean等等。
## 构造tensor变量
- 有0维，1,2,3维度等  
    ```python
    torch.tensor(1.0)   #构造一个0维的数据是1.0的tensor, print之后是tensor(1.0)
    torch.FloatTensor(2,3,4)    #构造3维元素是float类型的tensor，推荐使用
    torch.IntTensor(2,3,4)  #与上面类似，元素类型是int， **注意这两种构造方法未初始化，会随机赋值，所以构造完后，一定要注意给初始化，再使用！**
    ```
- torch变量的默认类型：FloatTensor
    ```python
    torch.Tensor(2,3,4)
    torch.FloatTensor(2,3,4)
    torch.tensor(1)
    torch.tensor([1,2,3])   # 这种情况特殊，当是列表或者数组时，其中是整形，那么该类型将是torch.LongTensor；其他即使输入的是整数，也会转化为floattensor。
    ```
- .type() 和 type() 
    ```python
    a = torch.tensor([1,2,3]).type()
    a.type()  #查看类型，能够给出tensor具体类型，这里的输出就是：torch.FloatTensor
    type(a) # 这个type是Python 内置的函数，用于判别各种类型的，所以它只能给出是：torch.tensor ，不能给出tensor的具体类型
    ```
## tensor 随机初始化
 - 均匀采样
        - rand: 在[0,1]之间均匀采样
        - randint: 自定义采样范围，只能均匀采样整数值
        - rand_like: 很有用，参照输入tensor的类型和shape来均匀采样，但是对randint类型不行
        ```python
        a = torch.rand(3,4)
        b = torch.rand_like(a)
        c = torch.randint(1,10,[3,4])   #输入的是{3, 4}大小，在1-10中均匀采样得到的整数
        ```
  - 高斯分布采样
   - torch.randn(3,4) #在正态分布中随机采样12次，放到$3\times 4$的tensor中
   - torch.normal(mean, std)  #可以自己定义均值和方差
  - torch.full([2,3],4)  #将$2\times 3$tensor 都取为4
  - torch.arange(1,10) 和Python中的range功能相同
  - torch.linespace(1,10, step=10)   #不是每一步走10,而是[1,10]分为10个数，从1开始
  - torch.ones(3,4)
  - torch.zeros(2,3)
  - torch.eye(4,5)
  - torch.ones_like()   #和rand_like类似
  - **torch.randperm()**  #和numpy中的random.shuffle类似，都可以将顺序打乱。这个功能很有用，绝对会用到。 


## tensor 和 Tensor
- tensor 接受数据，不接受维度的size，Tensor反之。
    ```python
    torch.tensor([2,3])  #接受的是一个具体的列表或者数据 
    torch.Tensor(2,3)  #此处接受的是一个2维大小为{2,3}所谓矩阵
    ```

## 和numpy的转换
- torch.from_numpy() , .numpy()
    ```python
    a = np.array([2, 3.3])
    a = torch.from_numpy(a)
    b = a.numpy()
    ```
## 和list的转换
- torch.tensor()， .tolist()
    ```python
    a = torch.tensor([1,2,3])
    b = a.tolist()
    ```
- 注意list()和.tolist()是不同的，前者同样会转化为一个list，但是元素类型是0维度的tensor。
    ```python
    b = list(a)
    print(b)
    output: [tensor(1), tensor(2), tensor(3)]
    ```

## tensor 的维度和tensor 的大小
- size(), shape, dim()
    - shape 是个属性，给出的是tensor的大小，可以做切片获取某一维的大小；
    - size()是个成员方法，和shape功能相同，也可以做切片获取tensor某一维的大小
    - dim(), 获取的是维度
    ```python
    a = torch.Tensor(3,4)  #会得到随机初始化的一个2维$3\times 3$的tensor 矩阵。
    a.shape   # torch.Size类型，就当一个列表使用即可。
    a.shape[0] #取第一维的大小
    a.size()  #torch.Size类型，功能和shape一样，自然类型相同
    a.size(0) #同样可以切片，不过实际上是对函数size()传参
    a.dim() #给出维度，类型是**int**
    ```

## 索引和切片和掩码选取
- 索引  
    ```python
    a = torch.rand(2,3,4,5)
    a[0]   #选取的是第一维中第一个的数据
    a[0].shape   # 结果是 torch.Size([3,4,5])
    a[0,2]  # 取第一维的第一个，第二维的第3个 的矩阵
    a[0,2,1,3]  #结果不是tensor处的数值，而是该数值的0维tensor
    ```
- 切片
切片方式和Python、numpy 类似。  
这里特别提示一个隔行和隔元素切片  
    ```python
    a = torch.rand(2,3,4)
    a[:, 0:3:2, 0:4:2]   #在2个3x4的矩阵中，从第一行开始，每隔一行取一行，在所取的行中，从第一个元素开始，每隔一个元素，取一个
    a[:,::2,::2]  #和上面的功能相同，算是简写，更方便
    ```
- mask 选取
    ```python
    a = torch.randn(3,4)
    mask = a.ge(0.5)    #选取元素中大于0.5，得到的是个掩码矩阵，a元素>0.5处，对应mask元素为True，否则为False。这和pandas中的处理方式一样
    torch.masked_select(x, mask)  #得到的是一个1维的tensor向量，就是将a矩阵打平，只选取其中大于0.5的元素，比如其中可能的一个结果是tensor([0.9138, 0.8501, 0.8648, 0.5103, 0.7447, 0.7549])
    ```

## 维度变换
- view / reshape
    - view : 将数据维度重置，但是要保证重置前后元素个数相同，这在逻辑上要保证的
    ```python
    a = randn(4,1,28,28)
    b = a.view(4, 28*28)
    c = a.reshape(4, 28*28)  #和view相同，将torch.Size([4,1,28,28])变为torch.Size([4,756])
    ```
- squeeze / unsqueeze
    - unsqueeze : 维度扩张
    ```python
    a = torch.rand(3,4,5).shape # 输出torch.Size([3,4,5])
    a.unsqueeze(0).shape # 输出 torch.Size([1,3,4,5])，其中0的意思是在第一个维度之前插入，现在第一个维度是3，所以如此
    a.unsqueeze(1).shape  # 输出 torch.Size([3,1,4,5])，在第二个维度之前插入
    a.unsqueeze(-1).shape # 输出 torch.Size([3,4,5,1])，在最后一个维度之后插入
    a.unsqueeze(-2)  # 在倒数第2个之后插入，也就是最后一个之前插入，正负号还是有规则的
    ```
    - squeeze
    只能对是1的维度进行压缩，将维度减少  
    ```python
    a = torch.rand(1,2,1,3)
    a.squeeze(0).shape   # torch.Size([2,1,3])
    a.squeeze(2).shape   # torch.Size([1,2,3])
    a.squeeze()  # 会将维度是1的都压缩 ，torch.Size([2,3])
    ```
- expand : 维度扩张，是将对应维度上的为1的进行扩张，扩张到多少，就会重复复制几次元素，相当于numpy中的广播。 
    ```python
    a = torch.rand(1,2,1,3)
    a.expand(3,2,4,3).shape  # torch.Size([3,2,4,3])
    a.expand(3,-1,4,-1)   # 和上面功能相同，只是-1表示该维度上不变，用于简写
    ```
- 广播机制
和numpy中的广播机制是一样的，要符合广播机制的规则： 从后向前匹配，匹配不是相等，就是后者的维度是1，才能广播. 还有一种情况，就是要后者也可以是0维，也一样可以加。还有一点，就是前后两者的类型必须相同，否则报错。
    ```python
    a = torch.randn(2,3,4)
    b = a + torch.tensor(1.)
    b = a + torch.tensor([1.])
    b = a + torch.tensor([1])  #报错，因为后者类型是torch.LongTensor，类型不匹配。
    ```

- 矩阵转置
    - .t : 用于二维的转置，高纬不可用
    - .transpose()
    - .permute()

- 分割和拼接
    - cat :拼接，要指定dim
    ```python
    a = torch.rand(3,4)
    b = torch.randn(4,4)
    torch.cat([a,b], dim=0)  # 按照第一维度拼接
    ```
    - stack

    - split 和 trunk

- 数学运算
    - +、-、*、/ : 这4种运算都使用了**广播机制，所谓广播机制，也就是对张量进行逐元素操作，这和矩阵的乘法是不同的，矩阵乘法是做线性变换，矩阵也没有除法。这里的*和/是数乘，/也是数乘的一种** 
    - 逐元素操作的加减乘除，有两种形式实现，一个是调用torch对应的函数，一个是用已经被重载的运算符。
    ```python
    a = torch.full([2,3], 4)
    b = torch.tensor([1., 2., 4.])
    c = torch.tensor(2.)
    a + b
    a - b
    a / b
    a * b
    b / a
    b - a
    c / a
    torch.add(a,b)
    torch.sub(a-b)
    torch.mul(a,b)
    torch.div(a,b)
    ```
    - 幂次方，指数，对数（以e为底）： 都是逐元素操作的
    ```python
    a = torch.fll([2,3],4)
    a.pow(2)   # 2次方，.pow 重载的运算符：**
    a**2
    a.sqrt()
    b = torch.exp(a)
    b = torch.log(a)  #默认以e为底，也可以是其他的
    ```

    - 张量（矩阵）的乘法
        - torch.mm :只对二维的可以使用，矩阵相乘
        - torch.matmul : 用的最多，可以更高维度使用，其重载的运算符是：@
        ```python
        a = torch.full([2,3], 4)
        b = torch.full([3,2], 2)
        c = torch.full([3,2], 3)
        torch.mm(a,b)
        torch.matmul(a,b)
        torch.matmul(a,c.t())
        a@b
        ```
        - **2维以上的矩阵乘法**
        用torch.matmul ，它默认会对后面两维进行矩阵的乘法运算，前后的维度保持不变，实际上可以认为是对多个矩阵进行并行运算。
        ```python
        a = torch.rand(4,3,28,64)   #假设这里4是batch中样本数，3是图片通道，(28,64)是图片像素，现在线性变换w是(64, 28) ，每个通道共享参数，我们进行矩阵相乘的时候，希望对batch中每一个样本，样本中每一个通道，都与同一个参数进行相乘。这里面的每一个就体现在**广播**
        w = torch.rand(64,28)  # 为了实现上面的功能，w要进行广播，会扩充为W.shape=(4,3,28,64),然后每一个w和a的后两维度（像素矩阵）相乘。就实现了。torch就是按照这个思想设计的
        torch.matmul(a, b).shape   # torch.Size([4,3,28,28])
        ```
    - clamp、min、max、norm、median 、prod
    ```python
    a = torch.rand(3,4)
    a.min()
    a.max()
    a.norm(2)  #给出的是2范数，很有用，可以查看梯度的2范数，来检测时候梯度会爆炸或者消弭
    a.median()
    a.clamp(0.5)  # 将<0.5的都调整为0.5，否则不变，可以对梯度进行截取，避免过大，过小
    a.clamp(0, 0.8) #调整的范围是(0,0.8)， 小于0的为0，大于0.8的为0.8。
    a.prod()  #所有元素连乘
    a.prod(dim=1)   #对每一行的元素连乘，其shape是torch.Size([3])
    ```
    clamp：钳子，夹紧，锁住的意思，这里就是将数据调整在一定的范围之内。
    **当指定维度的时候，比如a.max(dim=1),那么返回的是一个torch.return_types.max类型，包括两部分，一种一部分是每一行的最大值，另一部分是每个最大值在每一行的索引，要想取得每一行最大值，可以使用a.max(dim=1)[0]，相应的方法可以取得索引**

    - argmax、argmin和dim，keepdim
        - argmax 和 argmin 返回最大值和最小值索引的位置，这在找最后分类中，概率最大的索引很重要，索引位置和分类类别相对应，那么找到了索引位置，也就找到了分类。
        ```python
        a = rand(3,4)   # 见下面的分析
        a.argmax()
        a.argmax(dim=1)  # 就是返回每一行中的最大值,其shape是torch.Size([3])
        a.argmax(dim=1, keepdim=True) # keepdim 能够保存其他的维度，此shape是torch.Size([3,4])
        a.norm(2, dim=1) #求每一行的2范数
        ```
        a.argmax() : 将整个矩阵拉平，找到的最大值所在的索引位置，但是对于最后的数据，a的第一维一般认为是batch中的样本数，第二维认为是每个样本的所预测出各类别的概率。所以这种找不出，每个样本所预测的概率，要在每行中找最大值，也就是在dim=1中找最大值，这里有个容易疑惑的地方，那就是，在每行中找，不应该是dim=0么？其实是在每一个样本的每一列中找最大值，要是不容易理解，可以这样记住：**在某一行中找还是在某一列中找最大值，是容易判别的，以在每一行中找，那么每一行中对应的元素的个数是dim=0的个数，还是dim=1的个数，是哪一个就选哪一个**
    
    - topk ： 取最大的前几个
    ```python
    a = torch.rand(3,4)
    a.topk(3, dim=1)   # 取每一行前3个最大的
    a.topk(3, dim=1, largest=False)  # 取每一行最小的3个
    ```
    - compare : >, <, >=, !=, ==
    其中>号也用torch.ge()，== 也用torch.eq(a, b)
    
