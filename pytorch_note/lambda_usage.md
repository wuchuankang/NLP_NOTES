# lambda 函数
匿名函数，作用是简单的语句实现，配合filter 和 map， 能够很简洁的实现某些功能。
```python
a = [2,3,4,5,5,6,7,7,8]
b = list(filter(lambda x: x>4, a))
c = list(map(lambda x: x**2, a))
```
第二个参数必须是一个可迭代的类型，这两个函数将可迭代的，每一个元素一一放到前面的lambda函数中，然后输出。
