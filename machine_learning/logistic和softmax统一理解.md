## Logistic 和 softmax 统一理解
    我们已经知道，最大似然函数和交叉熵本质上是一回事，所以对于概率模型，以后直接考虑用交叉熵来构造目标函数就可以了。  
* 对于多分类问题，一般用softmax进行概率转化后，再用交叉熵来构造目标函数(当然也可以用多分类svm损失)，比如假设第i个样本$(x_i,y_i)$，类别用one-hot来编码，  
$$
W^T\bold x = 
\left[
\begin{matrix}
f_1\\
        f_2\\
        ...\\
        f_n\\
        \end{matrix}
\right] \tag{1}
$$
应用softmax之后，概率转化为：
$$
\left[
\begin{matrix}
\frac{e^{f_1}}{\sum e^{f_i}}\\
        \frac{e^{f_2}}{\sum e^{f_i}}\\
        ...\\
        \frac{e^{f_n}}{\sum e^{f_i}}\\
        \end{matrix}
\right] \tag{2}
$$
而真实的y的概率分布可以用one-hot来表示，即只在第$y_i$位置处为1，其他位置为0，那么应用交叉熵就是：
$$
-\log  \frac{e^{f_{y_i}}}{\sum e^{f_i}}    \tag{3}
$$
* 对于二分类问题，由于我们只要求出来为正类别的概率$p$，那么负类别可以$1-p$，为了节省参数，那么$W$只要是$n\times 1$即可，而不必要是$n \times 2$，我们用$\theta$来代替$W$，那么$\theta^T\bold x$假定是正类别的得分，怎么转化为概率呢？正好sigmoid满足这个特性：
$$
\sigma (z) = \frac{1}{1+e^{-z}}  \tag{4}
$$
那么预测的概率分布就是：
$$
\left[
\begin{matrix}
\frac{1}{1+e^{-z}}\\
\frac{e^{-z}}{1+e^{-z}}
\end{matrix}
\right] = 
\left[
\begin{matrix}
p_1\\
p_2
\end{matrix}
\right]
$$
其真是的概率分布表达成one-hot的形式就是：
$$
\left[
\begin{matrix}
y_i\\
1-y_i\end{matrix}
\right]
$$
所以其交叉熵就是：
$$
-(y_i\log p_1 +(1-y_i)\log p_2)   \tag{5}
$$
* 所以说，Logistic回归就是softmax回归的在二分类时候的特殊形式，反过来说，就是推广。
* 其实，我们可以看到，softmax对一个向量的各元素进行压缩，sigmoid对一个元素进行压缩，所以softmax和sigmoid是对应的。
* softmax名字的由来：
