## pytorch 学习策略 
学习策略主要是学习率如何更变，在 torch/optim 包内，不仅有各种优化器（参数更新策略），比如sgd, adam等，还有一个 lr_scheduler.py 模块，其中定义了学习率如何变化的策略。  
```python
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
```
其中 _LRScheduler 是基类， LambdaLR 继承 _LRScheduler.

在 huggingface/transformers 中实现了 学习率预热 的方法，优化器使用 AdamW:
```python
from transformers.optimization import ConstantLRSchedule, WarmupConstantSchedule,
                                    WarmupCosineSchedule, WarmupLinearSchedule
from transformers.optimization import AdamW 
```
其中 WarmupLinearSchedule 方法简单：  
$$
if \ \  step < warm\_step: \\ 
\lambda_{lr} = \frac {step} {warm\_step}   \\
if \ \ step > warm\_step: \\ 
\lambda_{lr} = \frac {total\_step -step} {total\_step - warm\_step}
$$
最终的学习率就是：
$$
lr = lr_{initial} * \lambda_{lr}
$$
所以，我们要给两个参数，一个是初始的 $lr_{initial}$ ，另外一个就是 预热的步 $warm_{step}$。   
而这些预热方法都继承了 LambdaLR ，看一下 WarmupLinearSchedule 的代码：
```python
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
```
再看一下 LambdaLR 的源代码：  

```python

class LambdaLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        self.last_epoch = last_epoch
        super(LambdaLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

```
在 LambdaLR 中定义了 get_lr ，它调用了 WarmupLinearSchedule 中的 lr_lambda 函数，实现了对 lr 的更新。  
**要注意的是，lr_lambda 函数这里变成了 self.last_epoch， 也就是说不是依据step ,而是依据epoch 实现预热，那么是否和 WarmupLinearSchedule 中用 step 相违背呢？**，这个等我们看完 _LRScheduler 的源码实现后再说。  

```python
class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

```
当实例化 scheduler =  WarmupLinearSchedule(...) 后，因为继承的关系，也就实例化了 LambdaLR 和 _LRScheduler，实例化了 _LRScheduler,  因为给的参数是last_epoch = -1，  所以 self.last_epoch = 0； 而我们每次更新 lr的时候，使用的是:
```python
scheduler.step()
```
也就是调用了爷爷类 _LRScheduler 中的 step() 函数， 而 step() 函数调用的是父类 LambdaLR 中的 get_lr()， 其中每调用一次 scheduler.step() ，那么 self.last_epoch 就会 +1，所以现在可以回答上面用粗体说明的问题了。要实现依据 step 预热的方法， 将 scheduler.step() 放到以步迭代的循环中就可以了，比如：
```python
for epoch in range(10):
    for i, batch in enumerate(dataloader):
        ...
        ...
        optimizer.step()
        scheduler.step()
        ...
```

**注意 optimizer.step() 更新的是 学习参数W， b， 而 scheduler.step() 更新的是lr**，在pytorch中 optimizer.step()更新要放在前，否则会报错。但是这样，第一次使用的 Qulr 就是 $lr_{initial}$了，似乎不合理，为什么这样还没搞清楚。
