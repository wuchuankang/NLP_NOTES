"""
用opencc将文件内容从繁体转化为简体来作为例子
如果不使用批处理，那么要在bash中为某一个文件夹下的每一个问价都执行一下语句
opencc -i input_file -o output_file -c 't2s.json'
批处理关键的核心是将命令用字符串表示，然后用os.system()来实现命令
"""
import os

def t_to_s(read_path, save_path):
    """t_to_s

    :param read_path: 读取文件的位置，可以使用相对路径，所谓相对路径，是相对与本脚本bat.py而言的
    :param save_path: 处理后文件的保存位置
    """
    filelist = os.listdir(read_path)    # 读取文件下的各个文件名称
    #print(filelist)

    for i in range(len(filelist)):  #对每一个文件进行遍历处理
        path_in = os.path.join(read_path, filelist[i])
        path_out = os.path.join(save_path, filelist[i]+'_s')
        command = '/usr/bin/opencc'+' '+ '-i' + ' ' + path_in + ' ' + '-o' + ' ' + path_out + ' ' + '-c' + ' ' + 't2s.json'     #将命令已字符串的形式表示出来
        os.system(command)  # 用os.system 来实现上面用字符串表达的命令



if __name__=='__main__':
    read_path = './zhwiki200/AA'
    save_path = './zhwiki200/BB'
    t_to_s(read_path, save_path)

