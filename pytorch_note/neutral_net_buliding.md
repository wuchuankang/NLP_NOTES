## 神经网络搭建

### 模型参数注册

- 一个神经网络模型可以有多可网络模块组成，每个网络模块都继承于 torch.nn.Module ，要想将模块的每一层都注册到模块参数中，用于模型训练(参数更新)，
    就要在 __init__() 中搭建的层使用 nn.module ，例子如下：
    ```python
    import torch as t
    import torch.nn as nn
    class T(nn.Module):
        def __init__(self, d=2, v=3):
            super().__init__()
            self.linear = nn.Linear(d, v)
            self.conv = nn.conv1d(1,2,3)
            self.dropout = nn.Dropout(0.1)
    a = T()
    for name, p in a.named_parameters():
        print(name, p)
    ```
    结果如下:

    ```python
    linear.weight Parameter containing:
    tensor([[-0.1456,  0.4381],
            [-0.6750, -0.3501],
            [-0.2352, -0.1397]], requires_grad=True)
    linear.bias Parameter containing:
    tensor([ 0.3160, -0.0950,  0.4168], requires_grad=True)
    conv.weight Parameter containing:
    tensor([[[-0.0093, -0.2881, -0.5286]],

            [[ 0.4936, -0.5096,  0.0219]]], requires_grad=True)
    conv.bias Parameter containing:
    tensor([-0.0424,  0.1792], requires_grad=True)
    ```
    注：  
    - 上面 self.dropout 因为没有参数，所以不在 model.named_parameters() 中
    - self.dropout = t.nn.functional.dropout() 来替换上面的行不行？ 答案是：最好不要这样，因为训练模式和测试模式不同，训练的时候使用 dropout ，但是测试的时候不使用 dropout，而我们想切换到测试模式，只要使用 model.eval() 即可，而使用 functional.dropout() ，则无法切换。  


- 如果网络中存在多个相同的层要重复使用，这里所说的相同的层，不光是层的名字相同，还有参数的规模大小相同，只是在内存中存储位置不同(如果这也相同，就不对了)。如何搭建？  

    - 从上面的叙述中，最明显的方法就是使用 copy.deepcopy() 函数，举例如下：
    ```python
    import copy
    def clones(module, n):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

    class T(nn.Module):
        def __init__(self, d=2, v=3, n=2):
            super().__init__()
            self.linears = clones(nn.Linear(d,v), n)
            self.drop = nn.Dropout(0.1)
    a = T()
    for name, p in a.named_parameters():
        print(name, p)
    ```
    结果是：  

    ```python
    linears.0.weight Parameter containing:
    tensor([[-0.2627,  0.0833],
            [ 0.3246, -0.0546],
            [-0.3258,  0.5900]], requires_grad=True)
    linears.0.bias Parameter containing:
    tensor([ 0.4920,  0.1770, -0.3272], requires_grad=True)
    linears.1.weight Parameter containing:
    tensor([[-0.2627,  0.0833],
            [ 0.3246, -0.0546],
            [-0.3258,  0.5900]], requires_grad=True)
    linears.1.bias Parameter containing:
    tensor([ 0.4920,  0.1770, -0.3272], requires_grad=True)
    ```

    - 另一种是使用列表，然后添加到 nn.ModuleList() 或者解包到 nn.sequential() 中，举例如下：
    ```python
    class T(nn.Module):
        def __init__(self, d=2, v=3, n=2):
            super().__init__()
            module_list = []
            for _  in range(n):
                module_list.append(nn.Linear(d, v))
            self.linears = nn.ModuleList(module_list)
            # 或者使用解包
            # self.linears = nn.Sequential(*module_list)
            self.drop = nn.Dropout(0.1)
    a = T()
    for name, p in a.named_parameters():
        print(name, p)
    ```
    结果是和上面相似的结果，只是参数数字不同，因为每一次都是随机初始化


### nn.ModuleList VS nn.Sequential()

- 这两者从名字可以看出来，前置是一个 列表的容器 ，后者是一个序列的容器，nn.module_list()接受的是一个列表，这个列表中的层或者模块在 forward 函数中的执行顺序有自己定，每一层可以像使用列表一样调用，上面的例子，我们缺少了 forward() 函数，可以在forward 中 self.linear[1] 来使用 linear.1
- nn.Sequential()，接受的是一个一个层或模块，所以要将列表给解包；它是有执行顺序的，顺序就是参数位置的顺序，作为前面参数的层或模块先执行，依次类推。因为这个特性， Sequential 内部已经定义好了 forward 函数（否则也不能实现依次执行），所以不用自己实现(这里不用自己实现，是说 nn.Sequential()单独成为模块，后面举例说明)；当然，这么看来上面用 nn.Sequential(*module_list)定义的模块就不对了，因为不可能连续矩阵乘两个相同的参数矩阵，维度不一致。
举例说明：
    ```python
    model = nn.Sequential(nn.Linear(2,3), nn.Linear(3,4))
    x = t.ones(2, 2)
    print(model(x))

    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.linears = nn.Sequential(nn.Linear(2,3), nn.Linear(3,4))

        def forward(self, x):
            return self.linears(x)
    a = T()
    print(a(x))
    print(a.linears(x))
    ```
    结果是：  
    ```python
    tensor([[-0.9790, -0.1650,  0.4554, -1.0907],
            [-0.9790, -0.1650,  0.4554, -1.0907]], grad_fn=<AddmmBackward>)
    tensor([[ 0.0671,  0.2037,  0.3092, -0.5926],
            [ 0.0671,  0.2037,  0.3092, -0.5926]], grad_fn=<AddmmBackward>)
    tensor([[ 0.0671,  0.2037,  0.3092, -0.5926],
            [ 0.0671,  0.2037,  0.3092, -0.5926]], grad_fn=<AddmmBackward>)
    ```

    上面的 model 是 nn.Sequential() 单独形成的，可以直接使用 model(x) ，因为forward 函数内部已经定义；或者如果没有实现 forward 函数，那么就是能使用 a.linears(x) 来实现， 要想使用 a(x) ，那么必须要实现 forward 。

- 模型的参数都在 model.named_parameters() 和 model.parameters() 中，模型的层或者模块在哪里呢？为什么上面可以通过 a.linears(x) 来调用linears 呢？  
同样，层或者模块(具备多个层) 都存在 model.named_modules() 和 model.modules() 中。但是这些模块既然是 实例 model 的属性，那么自然可以像普通的属性一样调用。并且像 nn.ModuleList()和 nn.Sequential() 中定义的，还可以像列表一样的调用使用。

- 网络模块如何累加？  

    这里说的意思是：对于复杂的网络，大多具有层次，我们可以通过把每一个层次的模块先单独建立起来，然后在组合在一起，下面给出一个简单的例子：
    ```python
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4,4)

        def forward(self, x):
            return self.linear(x)

    class T(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.linears = nn.Sequential(nn.Linear(2,3), nn.Linear(3,4))
            self.model = model

        def forward(self, x):
            return self.model(self.linears(x))
    b = M()
    a = T(b)
    print(a)
    ```
    结果是：

    ```python
    T(
      (linears): Sequential(
        (0): Linear(in_features=2, out_features=3, bias=True)
        (1): Linear(in_features=3, out_features=4, bias=True)
      )
      (model): M(
        (linear): Linear(in_features=4, out_features=4, bias=True)
      )
    )
    ```
- torch 中优化函数的使用  

    torch.optim 的各种优化类(optim.SGD、optim.Adam 等等) 构造函数__init__的模式是相同的，第一个是 参数的tensor或者是dict 形式，其他是优化选项：学习率、衰减率等等 。例子：
    ```python
    optim.SGD([{'params': model.base.parameters()},{'parms': model.classifier.parameters(), 'lr': 1e-3}], lr=1e-2, momentum=0.9)
    ```
    上面使用在model的base模块参数学习率没给，则使用外部的 lr=1e-2，而 classifier 模块使用的是 lr=1e-3。  
    更加常见的是，model 所有的模块参数使用相同的学习率：

    ```python
    optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.99))
    ```

- 在预热优化模型中，虽然各个层的学习参数的lr相同，但是lr是不断变化的，那么如何处理？ 

    优化器一旦初始化，就会把模型的参数存在 optim 的方法 param_groups 中，而 param_groups 是一个列表，元素是字典，字典中不光有 'params' 为关键字的参数，还有各个优化参数
    ```python
    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.linears = nn.Sequential(nn.Linear(2,3), nn.Linear(3,4))
            self.linear = nn.Linear(4,4)

        def forward(self, x):
            return self.linear(self.linears(x))
    a = T()

    optimizer = optim.Adam(a.parameters(), lr=1e-3)
    for group in optimizer.param_groups:
        print(group)
    print(optimizer.param_groups)
    ```
    结果是：  
    ```python
    {'params': [Parameter containing:
    tensor([[-0.4121, -0.2686],
            [-0.1014,  0.6734],
            [ 0.1920, -0.3846]], requires_grad=True), Parameter containing:
    tensor([0.2142, 0.5773, 0.0440], requires_grad=True), Parameter containing:
    tensor([[-0.4798,  0.4054, -0.5562],
            [-0.4894, -0.3493, -0.2682],
            [ 0.1913,  0.3688, -0.3334],
            [ 0.5355,  0.3446,  0.2861]], requires_grad=True), Parameter containing:
    tensor([ 0.3340, -0.0688, -0.0620, -0.3397], requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False}


    [{'params': [Parameter containing:
    tensor([[-0.5724,  0.2452],
            [ 0.2994,  0.3754],
            [-0.4384, -0.0547]], requires_grad=True), Parameter containing:
    tensor([0.1997, 0.4676, 0.6281], requires_grad=True), Parameter containing:
    tensor([[ 0.2288,  0.5056,  0.3609],
            [-0.5405, -0.3258, -0.2092],
            [-0.2337,  0.0926,  0.4641],
            [ 0.4418, -0.1646, -0.3612]], requires_grad=True), Parameter containing:
    tensor([-0.5483,  0.1433,  0.4261,  0.1951], requires_grad=True), Parameter containing:
    tensor([[ 0.1483, -0.1809,  0.4746, -0.4655],
            [ 0.4534,  0.1762,  0.0598, -0.0116],
            [-0.1364, -0.2698,  0.1886,  0.1932],
            [ 0.0573,  0.0883,  0.0195,  0.1002]], requires_grad=True), Parameter containing:
    tensor([-0.1092, -0.4762, -0.4504, -0.4206], requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False}]
    ```
    上面optimizer.param_groups 列表仅有一个元素，所以如此。那么会不会存在多个元素的情况？  
    这个和创建 优化器 有关系，从上面的例子可以看出，将模型 a的参数没有分层次的一起加入到优化器中，也即所有的参数使用同一套优化参数，自然只有一个。当参数创建如下：

    ```python
    optimizer = optim.Adam([{'params':a.linears.parameters()},{'params':a.linear.parameters(), 'lr'=1e-2}, lr=1e-3)
    print(optimizer.param_groups)
    ```
    结果是：
    ```python
    [{'params': [Parameter containing:
    tensor([[ 0.6445, -0.1027],
            [ 0.3591,  0.4638],
            [-0.3358, -0.5580]], requires_grad=True), Parameter containing:
    tensor([-0.5032, -0.5830,  0.4910], requires_grad=True), Parameter containing:
    tensor([[-0.2198, -0.3910, -0.2159],
            [-0.1562, -0.0292,  0.4589],
            [-0.3655, -0.5196, -0.2088],
            [ 0.4328,  0.0273, -0.2453]], requires_grad=True), Parameter containing:
    tensor([-0.4792,  0.3180,  0.1687, -0.1095], requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False}, {'params': [Parameter cont
    aining:
    tensor([[ 0.2525, -0.1185,  0.1518, -0.4299],
            [-0.1537, -0.2796, -0.3695,  0.1757],
            [-0.2220,  0.2686,  0.4030,  0.1438],
            [ 0.2409,  0.4663,  0.2762,  0.2060]], requires_grad=True), Parameter containing:
    tensor([-0.4105, -0.1045,  0.2088, -0.0043], requires_grad=True)], 'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False}]
    ```

    现在可以回答最初的问题，那就是所有学习参数虽然相同，但是随着 迭代步数 或者 epoch 会发生变化，怎么办？   
    ```python
    for group in optimizer.param_groups:
        group['lr'] = rate
    ```
    其中rate 就是随着步数或者 epoch 求出来的学习率。

