# pickle 模块

pickle 模块是Python中内置的包，可以序列化存储(二进制)一些数据，在机器学习中，常常需要把训练好的模型给存储起来，下次直接加载，可以使用 pickle 模块方便的存储。

- pickle.dump(obj, file) : 其中file是已经用序列化代开的，比如： open(file_path, 'wb')
- pickle.load(file) : 其中file是已经用序列化代开的，比如：open(file_path, 'rb')

```python
import pickle
dataList = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
dataDic = { 0: [1, 2, 3, 4],
            1: ('a', 'b'),
            2: {'c':'yes','d':'no'}}
 
#使用dump()将数据序列化到文件中
fw = open('dataFile.txt','wb')
# Pickle the list using the highest protocol available.
pickle.dump(dataList, fw, -1)
# Pickle dictionary using protocol 0.
pickle.dump(dataDic, fw)
fw.close()
 
#使用load()将数据从文件中序列化读出
fr = open('dataFile.txt','rb')
data1 = pickle.load(fr)
print(data1)
data2 = pickle.load(fr)
print(data2)
fr.close()
```
值得注意的是，可以对同一个文件多次dump，但是加载的时候，一次load也只能加载对应dump一次的数据。

# 补充关于open文件和close文件的问题

python 中经常会说，打开文件后，一定要关闭，为什么呢？这是因为，当你使用:
```python
f = open(file_path, 'w')
```
如果不f.close()，写在文件file_path中的内容不会马上写入到磁盘，还是在缓存中的，当cpu空闲的时候，才会写入到磁盘，而写入的时候，因为调度问题可能没法全部写入，从而可能会丢失文件，而调入f.close()，则会立刻写入到磁盘。  
按照上面的理解，那么如果打开的文件是只读的：
```python
f.open(file_path, 'r')
```
不关闭也应该没事，一般还是关闭的好。

