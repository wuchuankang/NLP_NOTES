## python 关于 import 的理解

    python 中有从路径上，有 绝对路径导入 和 相对路径导入； 从类型上来说，有包导入 和 模块导入

    以下面的程序结构为例子:
    ```python
    ├── dataset.py
    └── transformer
        ├── Constants.py
        ├── __init__.py
        ├── Optim.py
    ```
    其中 transformer 是一个包。  

- 包

    什么是包呢？在Python2 和 Python3.3 之前，包的定义是 文件夹 + __init__.py (在文件夹内部，内容可以为空)，  
    在 Python3.3 之后， 包不需要具备 __init__.py 文件也可以，但是一般都会写 __init__.py 文件，因为我们要在  
    其中定义很多 导入的模块

- 包导入和模块导入  

    先考虑在包内的模块相互调用， 比如要在 Optim.py 中调用 Constants.py 中的 BOS， Constants.py 中的内容是：
    ```python
    BOS = 'bos'
    ```
    那么 Optim.py 中可以写：
    ```python
    from Constants import BOS
    # import Constants.BOS as BOS 
    ```
    注释掉的语句会报错 ： Constants 不是一个包，可见只有包才可以使用 xxx.xxx 的形式来导入，比如：
    ```python
    import torch.nn.functional as F
    # from torch.nn import functional as F
    ```
    上面 torch 是包， nn 是包， functional 是一个模块，所以注释的语句也一样正确。



- 相对路径导入包  

    一般用绝对路径的少，这里只理解相对路径，在没有理解 相对路径 之前，总会遇到这样两个 error ：
    ```python
    ImportError: cannot import name 'Constants'
    ValueError: Attempted relative import beyond toplevel package
    ```
    在Python 导入包的路径中， '.' 、 '..'  和 '...' 分别代表~~当前路径、上一级路径、上两级路径，~~，这个是不准确的，应该是**当前包路径，上一级包路径，上2级包路径**，这里说是包路径的原因，因为某些原因，使得当前包不在是包的时候，那么用 '.' 会出现第一种错误， '..' 和 '...' 会出现第二种错误。这个很好理解，因为当前包和上一级不再是包，那么这几个点所表示的包路径就不对了。  
    什么情况会使得包失去包的意义？那就是该包下的一个模块作为函数入口，也就是运行某一模块，那么这个模块所在的包也就不再是包了。（至于为什么是这样，还待探讨）  

    现在举例解释：
    Optim.py 文件：
    ```python
    from . import Constants
    from .. import dataset
    ```
    运行 Optim.py 文件，会出现 ImportError: cannot import name 'Constants'，很容易解释，把 Optim.py 文件当做是函数入后，那么 transformer 就不再是一个包了，那么 '.' 也就没有了包对应了。 同时第二条语句会出现 ValueError: Attempted relative import beyond toplevel package， 同样，首先同级路径没有包，上级路径也就更没有了，所以会出现 beyond   

    __init__.py 文件：
    ```python
    from . import Constants
    from .Constants import BOS
    from .Optim import fun

    __all__ = ['Constants','BOS','fun']
    ```
    在 dataset.py 文件中调用：
    ```python
    from transformer import Constants
    from transformer import fun

    print(fun())
    print(Constants.BOS)
    ```
    运行 dataset.py 文件，结果正确，当然前提是 optim.py 文件中的 from .. import dataset 删除，因为一方面涉及到循环调用，另一方面， .. 正好对应的是 dataset.py 文件所在的包，但是当 dataset.py 当做函数入口，对应的这个包也就被破坏掉了。  
    __init__.py 文件 中的 . 对应的是 transformer 包，这个包不受 dataset.py 文件运行影响, 所以没事。  
    理解一下 from transformer import fun ，一直没说的是，当这样的语句出现时候，首先调用的是包transformer的 __init__.py 文件，而该文件内部我们已经将 fun 函数通过 from .Optim import fun 导入了， 所以可以直接使用。这也就是__init__.py 文件不是一个包必备的文件，我们还要写它的原因，因为能够简化调用。  
    __init__.py 文件中 __all__ 的作用，当使用 from transformer import *，就是将 __all__ 列表中的东西都导入到模块中，但是一般不这么使用，因为这样很容易造成命名的冲突。  

