# google colab usage
- 在 my dirve 下建立新的文件夹，然后选中该文件夹，点击new，选择more，选择 Google-colaboratory 就可以新建一个jupyter-notebook，可以选择配置是python3/python2，在执行代码程序的菜单项中选择运行时类型，来更改是cpu还是gpu。
- 想要查看自己使用的gpu信息，首先将jupyter-notebook，配置中运行时类型选择为gpu，否则使用：
    ```python
    !/opt/bin/nvidia-msi   
    ```
会显示没有该文件，只有配置了gpu，虚拟机上才会生成该文件。
- 想要显示和使用虚拟机上文件信息，需要挂载云盘：
    ```python
    from google.colab import drive
    drive.mount('/content/gdrive')
    ```
    **还有一个好处就是我们的云盘现在就在虚拟机/content/gdive下，我们将数据存放到云盘后，在程序中就可以读取云盘中的数据，就和使用自己电脑一样读取某个文件夹下的数据一样，同样计算的结果，也可以保存在云盘中，然后下载下来**
- 查看内存和cpu信息
    ```python
    !cat /proc/meminfo
    !cat /proc/cpuinfo
    ```
- 程序的运行
    - 直接使用jupyter-notebook，可以在上面编辑，也可以将本地的文件粘贴上去运行
    - 将本地的.py文件上传到my-drive 上，然后在jupyter-notebook 中运行：
    ```python
    !python xxx.py即可
    ```
    注意路径一定要写对，可以通过cd命令将路径切换到.py文件所在的文件夹下。
- 经测试发现，在jupyter-notebook中运行bash命令，如果是bash和系统的命令，不加!号一样可以，但是第三方软件必须加，比如Python。
- 查看gpu是否可用：
    ```python
    import torch
    torch.cuda.is_available()
    ```
