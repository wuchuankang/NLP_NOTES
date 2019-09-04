# pytorch usage
- .item : 当tensor中仅有一个元素时，可以将其转化为Python的数字，不论tensor是多少维的，这在将loss转化为float数值时经常用到，因为一开始初始化loss=0，所以后面计算loss的损失的时候，为了能够实现累加，后面计算出的loss要用item转化。  
- 对损失、精确到进行打印的时候，有格式的：
    ```python
    {:.0f}  # 表示保留整数位
    {:.3f}  # 表示输出小数点后3位   
    ```
- 计算正确率的时候
```python
correct = 0
pred = logits.ardmax(dim=1)
correct += pred.eq(target).sum().float().item()  #先要转化为float型，然后提出数值，因为整型相处不保留小数
accuracy = correct / len(data)
```
