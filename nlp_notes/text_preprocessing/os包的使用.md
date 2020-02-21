# os的使用
- os 对于路径的使用
    - os.listdir() 可以列出其参数文件夹下的所有文件名字，并以列表储存
    ```python
    import os
    filelist = os.listdir(/xxx)
    ```
    这个非常有用，在批处理的时候必然要用到

    - os.path.join(xxx, yyy), 将前后路径结合起来，批处理时，for循环每一个路径，也需要用到  
- os.system()的使用
用于批处理,将代码中的用字符串表达的命令作为该函数的参数，就相当于在shell中输入命令，这也是批处理的核心，例如opencc的使用，直接在shell中输入命令 ：  
opencc -i input_file -o output_file -c "t2s.json" 就可以将input_file 繁转简,然后放在output_file中，但是一个命令只能处理一个，如果是上百个，那就麻烦了。在批处理代码中该命令的实现：
``` python
import os

filepath = './zhwiki200/AA'
filepath_save = './zhwiki200/BB'
filelist = os.listdir(filepath)

for i in range(len(filelist)):
    path =os.path.join(filepath, filelist[i])
    save_path = os.path.join(filepath_save, filelist[i]+'_s')
    command = '/usr/bin/opencc' + ' '+ '-i'+' ' + path + ' '+ '-o'+' ' + save_path+ ' '+ '-c'+' ' + 't2s.json'
    os.system(command)
```

