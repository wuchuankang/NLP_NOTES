# pytorch 中 hook 的使用
之所以要可能用到hook，是因为在module中的forward()和loss.backward()中无法提取中间值，对与forward而言，没法提取某一层输出结果(如果想看某一层提取的特征的话)，backward计算结束后，对中间变量的梯度就清除了，也无法提取。这就要使用hook技术。    
- torch.register_hook()

- Module.register_forward_hook()

- Module.register_backward_hook()

# torch.register_hook 提取某一个属性是requires_grad=true的变量的导数

# module.register_forward_hook() 是提取某一个module的输入输出
先看一下定义在Module类中的代码：
```python
    def register_forward_pre_hook(self, hook):
        handle = hooks.RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle
```
将hook放到self._forward_hooks字典中，然后在__call__()中调用。
```python
    def __call__(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            result = hook(self, input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                input = result
        if torch._C._get_tracing_state():
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result
```
这里关键要注意的是，resigter_forward_hook 中的hook函数的参数具有固定的形式，也就是函数签名具有固定的形式：  

```python
hook(module, input, output) -> None
```
要说明的是不能有返回值，这可以从__call__函数中看出，当有返回值的时候，result = hook_result，那么就会将上面result=self.forward()的给覆盖了，那么前向传播就不准确了。  
参数说明： 
    - module是module.register_forward_hook(hook)最前面的module对象
    - input是该module的输入(因为module可以代表整个神经网络，也可以是代表神经网络中的某一层，当是整个神经网络，input就是最开始的输入，对应于实际就是批训练集，output就是最后一层的输出；当如果是某一层，那么input就是某一层的输入，output就是该层的输出，对应于实际就是激活后的输出，对于batchnorm和dropout则是经过各自运算后的输出)。

具体的例子：

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

hook_list = []
def hook(nodule, input, output):
    hook_list.append(input)
    #hook_list.append(output)

a = Net()
x = t.randn(1,3,28,28)
##################################################

handle = a.register_forward_hook(hook)  #a.layers.register_forward_hook(hook)
_ = a(x)
print(hook_list)
handle.remove()
####################################################

handle = a.layers[1].register_forward_hook(hook)
_ = a(x)
print(hook_list)
handle.remove()
```
这里要注意的
