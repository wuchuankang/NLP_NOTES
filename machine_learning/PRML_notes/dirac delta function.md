## Dirac delta function

本文主要参照[该网址](https://www.probabilitycourse.com/chapter4/4_3_2_delta_function.php)写成，该网址有不少值得学习的东西，值得浏览。

Dirac delta 函数经常作为一种桥梁的作用出现，它一方面使得离散变量具备了概率密度，从而使得离散变量和连续变量具有统一的表达形式；另一方面，它在一些推导过程中，起到转化的作用，具体例子下面会谈到。

### Dirac delta function 来源与性质

+ Dirac定义的来源

  考虑阶跃函数：
  $$
  \begin{equation}
  			  \hspace{50pt}
                u(x) = \left\{
                \begin{array}{l l}
                  1  &  \quad  x \geq 0 \\
                  0 &  \quad \text{otherwise}
                \end{array} \right.
  			  \hspace{50pt} (1.1)
              \end{equation}
  $$
  这个函数在$0$点处不连续，现在想找到一个函数来近似它，并且是连续可导的，要找这样的函数，自然从简单的线性函数入手，那么可以用下面的线性函数：
  $$
  \begin{equation}
               \nonumber u_{\alpha}(x) = \left\{
                \begin{array}{l l}
                  1  &  \quad   x > \frac{\alpha}{2} \\
                  \frac{1}{\alpha} (x+\frac{\alpha}{2})  &   \quad  -\frac{\alpha}{2} \leq x \leq \frac{\alpha}{2} \\
                  0 &  \quad x < -\frac{\alpha}{2}
                \end{array} \right.
              \end{equation}   \tag {1.2}
  $$
  其中$\alpha \gt 0​$，那么：
  $$
  u(x)=\lim_{\alpha \to 0} u_{\alpha}(x)=\lim_{\alpha \to 0} \frac{1}{\alpha} (x+\frac{\alpha}{2}) \tag{1.3}
  $$
  $u_{\alpha}(x)$的导数：
  $$
  \begin{equation}
               \nonumber \delta_{\alpha}(x)=\frac{ d u_{\alpha}(x)}{dx} = \left\{
                \begin{array}{l l}
                  \frac{1}{\alpha}  &  \quad  |x| < \frac{\alpha}{2}   \\
                  0 &  \quad |x| > \frac{\alpha}{2}
                \end{array} \right.
              \end{equation}  \tag{1.4}
  $$
  从而$u(x)$的导数可以用下面的极限表示：
  $$
   \nonumber \delta(x)=\frac{ d u(x)}{dx} =\lim_{\alpha \to 0} \frac{1}{\alpha}=+\infty \tag{1.5}
  $$
  注意，上面的$\alpha \to 0$应该是$\alpha \to 0^{+}$，因为$\alpha \gt 0$ ，所以极限应该是单侧趋近于$0$，从而：
  $$
  \begin{equation}
              \nonumber  \delta(x) = \left\{
                \begin{array}{l l}
                  \infty  &  \quad x=0  \\
                  0 &  \quad \text{otherwise}
                \end{array} \right.
              \end{equation}  \tag{1.6}
  $$
  上式就是Dirac delta函数，或者叫Dirac 函数。

  Dirac 函数的特点，可以看成是高斯函数对方差的极限：
  $$
  \delta(x) = \lim_{\alpha \to 0} \delta_{\alpha}=\lim_{\alpha \to 0} \frac{1}{|a|\sqrt{\pi}}e^{-({x}/{a})^2}  \tag{1.7}
  $$
  这种用高斯函数定义的形式，在贝叶斯模型关于高斯分布中会有用处，后面会讲到。

+ Dirac函数性质

  性质1：
  $$
  \int_{-\infty}^{\infty} g(x) \delta(x-x_0) dx = g(x_0)  \tag{2.1}
  $$
  这叫筛选(sifting)性质。

  证明如下：
  $$
  \begin{aligned}
  \int_{-\infty}^{\infty} g(x) \delta(x-x_0) dx
  &=\lim_{\alpha \to 0}\int_{x_0-\frac {\alpha}{2}}^{x_0+\frac {\alpha}{2}} g(x) \delta_{\alpha}(x-x_0) dx\\
  &=\lim_{\alpha \to 0}\int_{x_0-\frac {\alpha}{2}}^{x_0+\frac {\alpha}{2}} g(x) \frac {1}{\alpha} dx\\
  &=\lim_{\alpha \to 0} \alpha \frac {1}{\alpha} g(x_{\eta}) \ \ \ \ \ \ \ \ \ 中值定理，x_0-\frac {\alpha}{2} \le \eta \le x_0+\frac {\alpha}{2} \\
  &=g(x_0)   \ \ \ \ \ \ \ \ \  \lim_{\alpha \to 0} (x_0-\frac {\alpha}{2} \le \eta \le x_0+\frac {\alpha}{2})=x_0
  \end{aligned}   \tag{2.2}
  $$
  性质2：
  $$
  \int_{-\infty}^{\infty} \delta(x) dx = 1   \tag{2.3}
  $$
  $(2.3)$式的出现就自然而然了，只要令$g(x)=1$。

+ 从以上可以看出来，$\delta(x)$具备作为概率密度的特性：
  $$
  \delta(x) \ge0\\
  \int_{-\infty}^{\infty} \delta(x) dx = 1    \tag{3.1}
  $$
  由Dirac函数本身的性质，那么$(2.3)$就无需在正负无穷上积分为1，对任意的$\alpha >0$：
  $$
  \int_{-\alpha}^{\alpha} \delta(x) dx = 1   \tag{3.2}
  $$

### Dirac delta function 用于离散变量的表示

+ 对于离散变量$X$，取值范围$R_X=\{x_1,x_2,x_3,...\}$，概率质量函数(pmf)为$P_X(x_k)$，那么累积分布函数(cdf)可以写成：
  $$
  F_X(x)=\sum_{x_k \in R_X} P_X(x_k)u(x-x_k)   \tag{4.1}
  $$
  其中$u(x)$的定义见$(1.1)$式。

  则其概率密度(pdf):
  $$
  \begin{aligned}
  f_X(x)
  &=\frac{dF_X(x)}{dx}\\
  &=\sum_{x_k \in R_X} P_X(x_k)\frac{d}{dx} u(x-x_k)\\
  &=\sum_{x_k \in R_X} P_X(x_k)\delta(x-x_k)
  \end{aligned}  \tag{4.2}
  $$
  离散型概率密度表示出来后，那么求期望和方差用概率密度求和用概率质量求有区别么？没有，如果有，那么就不自恰了。

  例如求期望：
  $$
  \begin{aligned}
  EX 
  &=\int_{-\infty}^{\infty} xf_X(x)dx\\
  &=\int_{-\infty}^{\infty} x\sum_{x_k \in R_X} P_X(x_k)\delta(x-x_k)dx\\
  &=\sum_{x_k \in R_X} P_X(x_k) \int_{-\infty}^{\infty} x \delta(x-x_k)dx\\
  &=\sum_{x_k \in R_X} x_kP_X(x_k)
  \end{aligned}   \tag{4.3}
  $$
  可见是一致的。

### Dirac delta function 应用

+ 自然是将离散变量和连续型变量表达上统一起来了，这样可以用来表示混合变量，就是随机变量$X​$取值范围既有离散也有连续，概率表达可以用一个式子来表达，具体例子见[该网址](https://www.probabilitycourse.com/chapter4/4_3_2_delta_function.php)；
+ 用在经验分布上，参照《最大似然函数、交叉熵和经验分布的关系》；
+ 用在推导转化中，参照《贝叶斯logistic回归》。

