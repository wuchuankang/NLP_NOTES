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

## __dict__ 和 dir()
Python一切皆对象，那么各种变量就是各种类的实例(也就是对象)，这里面有各种继承关系，各个类和对象又有各自的属性，也有继承的属性；为了避免这种属性的混乱，需要一套属性的管理机制，这个就是靠__dict__属性来管理的。属性是分层管理的，对象的属性包括其类中定义的对象属性(self.xxx)，也包括从父类中继承的，要想从父类中继承属性，必须在__init__中调用父类的__init__);类的__dict__ 包括类的属性和类的方法，不包括继承中类的属性和方法，更不会包括子类的属性和方法。  
```python
class A(object):
    a = 2
    def __init__(self):
        self.b1 = 2

    def fun(self):
        print(self.b1)

class a(A):
    e = 5
    def __init__(self):
        super(a, self).__init__()    #如果没有它，则A中的self.b1 不会继承，就不会出现在a所定义对象的__dict__中
        c = 3    #注意，这个不是类的属性，只是__init__函数的一个临时变量
        self.d = 4

    def gun(self):
        print(self.d)

    def diss():
        print(A.c)

aa = A()
bb = a()
print(aa.__dict__)
print(A.__dict__)
print(bb.__dict__)
print(a.__dict__)
print(dir(bb))  # 将列出bb中所有的属性和方法，包括继承的方法和属性
```
