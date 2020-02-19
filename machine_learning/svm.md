## SVM
SVM 是 **二分类求解器！二分类！**
SVM 可以用作线性分类器，也可以用作非线性分类器，对于后者，是通过引进核函数，将**输入**数据映射到高维空间(甚至是无穷维)上，从而在无穷维上，**输出**是可分的。
训练集是 $\{(x_1,y_1),...,(x_m,y_m)\}$，其中$x_i \in R^n, y_i \in \{-1, 1\}$。  

### 模型推导
先考虑线性不可分的二分类问题，模型的分离超平面 $w^Tx+b=0$，决策函数是 $f(x) = sign(w^Tx+b)$。学习策略是软间隔最大化。  
我们从合页损失(hinge loss)的角度来考虑， SVM 的合页损失对应的目标函数如下：
$$
\min_{w,b}\sum_{i=1}^m \max(0, 1-y_i(w^Tx_i+b)) + \lambda ||w||^2 \tag{1}
$$
上面这个是结构风险，其意思是：当 $(x_i,y_i)$ 分类正确($y_i(w^Tx_i+b)>0$)且函数间隔$y_i(w^Tx_i+b)>1$时，损失才是0，否则损失就是 $1-y_i(w^Tx_i+b)$。  从公式$(1)$中，可以推导出线性支持向量机原始优化问题，或者说两者等价。推导如下：  
令
$$
\max (0，1-y_i(w^Tx_i+b)) = \xi_i  \\
\Rightarrow \min_{w,b}\sum_{i=1}^m \xi_i + \lambda ||w||^2 \tag{2}
$$
显而易见，$\xi_i>0$；当$1-y_i(w^Tx_i+b) > 0$，$\xi_i=y_i(w^Tx_i+b)$，当$1-y_i(w^Tx_i+b)<0$，$\xi_i=0$，可见$1-y_i(w^Tx_i+b) \le \xi_i$；   
令 $\lambda = \frac{1}{2C}$，公式$(2)$ 变为
$$
\min_{w,b}\frac{1}{C}(\frac{1}{2}||w||^2 + C\sum_{i=1}^m \xi_i)  \tag{3}\\
s.t. \quad 1-y_i(w^Tx_i+b) \le \xi_i \\
 \xi_i \ge 0, \quad i=1,2,...,m
$$
这个和下面的原始公式等价：
$$
\min_{w,b,\xi}(\frac{1}{2}||w||^2 + C\sum_{i=1}^m \xi_i)  \\
s.t. \quad y_i(w^Tx_i+b) \ge 1-\xi_i \\
 \xi_i \ge 0, \quad i=1,2,...,m   \tag{4}
$$

### 求解推导
#### 凸优化问题
参见《统计学习方法》$p_{116}、p_{120}-p_{121}$。  
公式$(4)$ 是一个凸优化问题，而凸优化这样的 **约束最优化问题**：
$$
\min_w f(w) \\
s.t. \quad g_i(w) \le 0， \quad i=1,...,k \\
h_i(w)=0，  \quad i=1,...,l
$$
**当$f(w)$和$g_i(w)$是连续可微的凸函数，$h_1(w)$是仿射函数的时，该约束最优化问题是凸优化问题。**
**当$f(w)$是二次函数，约束函数$g_i(w)$是仿射函数时，凸优化问题变成了凸二次规划问题。**
- 凸函数：对任意的$x,y \in R^n$，任意的$\alpha,\beta \in R$，且满足$\alpha+\beta=1, \alpha \ge 0, \beta \ge 0$，下面不等式成立：
    $$
    f(\alpha x + \beta y) \le \alpha f(x) + \beta f(y)
    $$
    典型的例子就是$f(x)=x^2$凸函数，还有更简单的$f(x)=x$也是，而$f(x)=-x^2$是凹函数。
- 仿射函数：就是一个线性函数，仿射变换就是线性变换，即如下的形式：$f(x)=a\dot x+b, a\in R^n, b\in R, x\in R^n$。

### 对偶问题
公式$(4)$是凸优化问题，在于优化函数是一个二次函数和线性函数的组合。  
通过 Lagrange 函数，可以将原始问题转化为对偶函数来解决问题，使用对偶函数**有2方面的好处**：
1. 不等式约束一直是优化问题中的难题，求解对偶问题可以将支持向量机原问题约束中的不等式约束转化为等式约束；
2. 可以在 SVM 中很自然的引进 **核函数**。支持向量机中用到了高维映射，但是映射函数的具体形式几乎完全不可确定，而求解对偶问题之后，可以使用核函数来解决这个问题。

上面2个好处将逐一在后面的推导中显示出来。  
为了避免书写公式的复杂性，参见《统计学习方法》$p_{127}$ 进行梳理。  
公式$(4)$如下：
$$
\min_{w,b,\xi}(\frac{1}{2}||w||^2 + C\sum_{i=1}^m \xi_i)  \\
s.t. \quad y_i(w^Tx_i+b) \ge 1-\xi_i \\
 \xi_i \ge 0, \quad i=1,2,...,m   \tag{4}
$$
通过 lagrange 乘子构建 lagrange 函数。对每一个不等式约束引进一个 lagrange 乘子 $\alpha_i \ge 0, \mu_i \ge 0$，构建 lagrange 函数$L(w,b,\xi,\alpha,\mu)$：
$$
L(w,b,\xi,\alpha,\mu)= \frac{1}{2}\lVert w \rVert^2+C\sum_i^m\xi_i-\sum_i^m(\alpha_i(y_i(w^Tx_i+b)-1+\xi_i))-\sum_i^m\mu_i\xi_i    \tag{5}
$$
原问题$(4)$式转为为 **极小极大**问题：
$$
\min_{w,b,\xi} \max_{\alpha,\mu}L(w,b,\xi,\alpha,\mu)   \tag{6}
$$
转化为对偶问题就是将 **极小极大** 写成 **极大极小** 问题：
$$
\max_{\alpha,\mu}\min_{w,b,\xi} L(w,b,\xi,\alpha,\mu)   \tag{7}
$$
**$最优解_{极小极大} \ge 最优解_{极大极小}$** ，可以从瘦死的骆驼比马大来简单的理解。  
**通过KKT条件，对偶问题得到的最优解就是原问题的最优解！！**

现在对公式$(6)$内部最小值先求导：
$$
\nabla_w L(w,b,\xi,\alpha,\mu) = 0  \\
\nabla_b L(w,b,\xi,\alpha,\mu) = 0 \\
\nabla_{\xi_{i}} L(w,b,\xi,\alpha,\mu) = 0
$$
结果得到：
$$
w=\sum_i^m \alpha_i y_i x_i \\
\sum_i^m \alpha_i y_i = 0 \\
C-\alpha_i-\mu_i=0 
$$
将上面带入到式$(7)$中，对偶问题就变成：
$$
\max_{\alpha} - \frac{1}{2}\sum_i^m \sum_j^m \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) + \sum_i^m \alpha_i \\  \tag{8}
s.t. \quad \sum_i^m \alpha_i y_i =0 \\
C-\alpha_i-\mu_i=0  \\
\alpha_i \ge 0 \\
\mu_i \ge 0
$$
**将后3项约束合写成一体**：
$$
0 \le \alpha_i \le C
$$
最终形式为：
$$
\max_{\alpha} - \frac{1}{2}\sum_i^m \sum_j^m \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) + \sum_i^m \alpha_i \\  \tag{9}
s.t. \quad \sum_i^m \alpha_i y_i =0 \\
0 \le \alpha_i \le C, \quad i=1,...,m
$$

**从式$(9)$中来分析转化为对偶问题后的3个优点**：
1. 将不等式约束$(4)$转化为等式$(9)$，其中$0 \le \alpha_i \le C$约束是很好处理的，因为是对$\alpha_i$进行了一刀切的约束，不像之前要根据样本来进行处理；  
2. 式$(9)$中有输入空间的内积 $(x_i \cdot x_j)$，当使用核技巧将输入空间映射到高纬的特征空间，然后求解目标函数的对偶函数，最后的结果恰好就是将 输入空间的内积 $(x_i \cdot x_j)$ 换成核函数 $K(x_i,x_j)$，即：
$$
\min_{\alpha} \frac{1}{2}\sum_i^m \sum_j^m \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_i^m \alpha_i \\
s.t. \quad \sum_i^m \alpha_i y_i =0 \\
0 \le \alpha_i \le C, \quad i=1,...,m
$$
其中核函数 $K(x, z)$ 定义如下：  
$X$是输入空间，$H$是特征空间，如果存在一个从 $X$ 到 $H$ 空间的映射
$$
\phi(x):X \rightarrow H
$$
使得对所有的$x,z \in X$，函数 $K(x,z)$都满足
$$
K(x,z) = \phi(x) \cdot \phi(z)
$$
$K(x,z)$ 就是核函数。
我们不需要知道映射$\phi(x)$具体是什么，只要知道映射的内积，也就是核函数，就可以在高维空间上使用核函数了。  
可以看出，通过转化为求解对偶函数，很容易引进映射到高维空间中使用核函数的SVM分类器。  

#### 分离超平面和决策函数
原始问题的分离超平面：
$$
w^T x + b = 0   \tag{10}
$$
通过求解对偶问题，再加上KKT条件， **对偶问题的最优解就是原始问题的最优解**  
通过KKT，可以得到
$$
w^* = \sum_i^m \alpha^* y_i x_i \\
b^* = y_i-\sum_i^m y_i \alpha^* (x_i \cdot x_j)
$$
其中$\alpha^*$是对偶问题最优解，$w^*,b^*$是原始问题的最优解。  
带入到$(10)$，得到分离以对偶问题下表示的超平面
$$
 \sum_i^m \alpha^* y_i (x \cdot x_i) + b^* = 0
$$
决策函数
$$
 f(x) = sign(\sum_i^m \alpha^* y_i (x \cdot x_i) + b^*)
$$

对于使用核函数的非线性 SVM，超平面
$$
 \sum_i^m \alpha^* y_i K(x, x_i) + b^* = 0
$$
决策函数
$$
 f(x) = sign (\sum_i^m \alpha^* y_i K(x, x_i) + b^*)
$$

#### 常用核函数
- 高斯核函数(gaussian kernel function)
    $$
    K(x,z) = \exp(- \frac{||x-z||^2}{2\sigma^2})
    $$
- 多项式核函数(polynomial kernel function)
    $$
    K(x,z)=(x \cdot z +1)^p
    $$
    p (整数)和$\sigma$是超参
    

#### 对偶问题的求解 
已将问题转化为求对偶问题，那么如何对下面约束优化问题进行求解呢？
$$
\min_{\alpha} \frac{1}{2}\sum_i^m \sum_j^m \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) - \sum_i^m \alpha_i \\  \tag{11}
s.t. \quad \sum_i^m \alpha_i y_i =0 \\
0 \le \alpha_i \le C, \quad i=1,...,m
$$
该问题是关于$\alpha$的二次规划问题，而求解二次规划问题，最好的方法是使用$SMO(sequential\, minimal \, optimization)$序列最小最优化方法。具体参见《统计学习方法》$p_{142}$。  
是否可以用梯度下降法呢？原则上是可以的，但是计算量大，收敛速度慢。

### 多分类 SVM 
我们开头就说了， SVM 是个二分类分类器，这里的多分类，其实是将 **合页损失** 应用到多分类中，一般是在深度学习中图片分类最后一层全连接后构造损失函数所用，其具体形式：
$$
L_i = \sum_{j\ne y_i}max(0, \delta + s_j - s_{y_i})
$$
意义是：对第$i$个样本而言，只有当该样本预测是$y_i$的得分$s_{y_i} \ge \delta + s_j$的时候($s_j$是该样本预测是其他类别的得分)，才没有损失，$\delta$是超参，用于控制损失的程度。  

### 分类求解器比较：朴素贝叶斯、logic、svm

- 朴素贝叶斯法：适合输入空间是离散的，可以二分，也可多分类
- logic 和 svm 是二分类器，输入空间可以是离散的，也可以是连续的
- 吴恩达的观点
    1. 如果Feature的数量很大，跟样本数量差不多，这时候选用LR或者是Linear Kernel的SVM  
    2. 如果Feature的数量比较小，样本数量一般，不算大也不算小，选用SVM+Gaussian Kernel  
    3. 如果Feature的数量比较小，而样本数量很多，需要手工添加一些feature变成第一种情况

- 当数据量非常大，完全跑不动SVM的时候，跑LR。
- 参照[SVM和logistic回归分别在什么情况下使用？](https://www.zhihu.com/question/21704547)
