## python 命令行解析模块 argparse

linux 命令是怎么构建的，自己写的Python脚本程序能否使用带参数的命令行来运行，是可以的，Python 标准库中使用的命令行解析模块是 argparse 。

### 位置参数 和 可选项参数

- 位置参数 ： 是必须要给的，这个像是程序中的没有给默认值的参数，这里的位置参数也可以有默认值，但是因为位置参数必须在运行的时候给定，所以这里给默认参数就没有必要，如果想像程序中的默认参数那样，可以不用给定，那么就要将该参数更变为 可选项参数。  

- 可选项参数 : 在命令运行事时，该参数可以不给定，不给定就使用默认的值；如果给定，后面要带上另外的参数来表示该可选参数的值，但当使用 action 来建立可选参数的时候，那么可以省略另外的参数；但当构建 可选参数时，使用了 required=True，则该可选蚕食不能省略。

- 注意可选参数使用 '-' 或 '--' 开头，前者默认是缩写，后者默认是全写，比如 Linux 命令中 -h 和 --help ；位置参数则不能带.
    ```python
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-bose', help='echo the commond you use')
    parser.add_argument('-gose', action='store_true')
    parser.add_argument('-tose', required=True)
    parser.add_argument('-sose', type=int, default=50)
    parser.add_argument('squrad', help='the **2', type=float)

    args = parser.parse_args()

    print(args.squrad)
    print(args.gose)
    print(args.sose)
    ```
    例子: 假设上面的程序名字是test.py，在 terminal 中输入：
    ```python
    python test.py -tose 1 0
    python test.py 0 -tose 1 
    python test.py 0 -bose 2 -tose 1 -sose 5 
    python test.py 0 -bose 2 -tose 1 -sose 5 -gose 
    ```

    解释: 上面第一二句是相同的，也就是说参数位置可以改变，一般吧位置参数放到命令参数的第一位， -tose 1 表示将 args.tose = 1（这个在实际使用中具体给定），因为tose required=ture，所以该参数在命令行中必须给；没有给出 -sose, 那么 args.sose=50 (默认值)； -gose 没有给，那么 args.gose=False，第4个命令给了，那么 args.gose=True


