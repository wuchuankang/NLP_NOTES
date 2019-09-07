# Python中有几个特殊方法是理解 pytorch 的关键
getattr 和 setattr 是Python builtin 方法。
## __getattr__ 和 getattr

__getattr__ 和 getattr 是用来获取对象的属性的，obj.attr 等价于obj.getattr(),当在该方法中找不到，则调用 obj.__getattr__()

## __setattr__ 和 setattr 

是用来更改属性的， obj.name = value 等价于 obj.__setattr__('name', value)，当没有定义该函数，则调用 obj.setattr()

## __getitem__ 和 __setitem__ 、 __len__
是用来定义容器类的，比如Python builtin 容器类数据结构list 、tuple和dict ，都有这3个方法，尤其是前两个方法:
- __getitem__ : obj[index] 等价于调用 obj.__getitem__(index)，**注意参数是index**
- __setitem__ ：obj[index] = value 等价于 obj.__setitem__(index, value)
- __len__ : 返回样本的数量， len(obj) 等价于调用 obj.__len__()

## __iter__ 和 __next__
可将类生成迭代器对象，这里提一下for遍历：
```python
a = [1,2,3,4]
for i in a:   #等价于 for i in iter(a)
    print(i)
```
只要是容器，可以通过iter函数转化为迭代器，for in 是个语法糖，可以认为是这样工作的，先判断被迭代对象是迭代器还是容器，是前者的话，就不用转了；是后者的话，就用iter()函数来转化为迭代器；否则就报错，不是可迭代对象。

## __call__
将类的对象当做是一个方法使用，只要将被调函数写在这里


