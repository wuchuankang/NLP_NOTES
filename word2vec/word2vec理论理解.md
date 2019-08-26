# word2vec 理论理解和确化
本文主要参照了[《word2vec中的数学原理详解》](https://www.cnblogs.com/peghoty/p/3857839.html)，但是文章中对负采样模型中关于目标函数理解的不够到位，后面会谈到。     
想要理解word2vec，要对神经网络要有清晰的理解，输出层主要用了 softmax 进行了概率化。
首先来说，word2vec使用的是概率模型，和n-gram从思想上是一样的，都是预测某个词的概率，n-gram 模型是用统计的方法预测n个连续词的概率，为了简化计算，会做出马尔科夫0阶或1阶或者2阶假设。而word2vec中，是对条件概率进行建模，cbow模型是求用词语w周边的词来预测w，也就是对$p(w|context(w))$，而skip-gram 模型则是对词w周边的词进行预测，即求$p(context(w)|w)$进行预测，具体如何实现，下面慢慢道来。  
要讲word2vec模型，则绕不开Benjio等人在《a neural probabilistic laguage model》中提出的语言模型（语言模型就是对词语的概率进行概率建模的模型），这是一个有监督问题，它是用词w的前n个词语（最多可做到n=5）来预测w。

## 神经概率语言模型
用神经网络对$p(w|context(w))$进行建模，其中$context(w)$是w的前n个词语。对于语料库C(corpus)建立词典D，D一般是C中不重复的单词，但也不绝对，比如可以有的人会在后续的处理中去除停用词，但不去除也可以，因为某些停用词，在某些场景中也是有意义的。我们对词典中每一个词用ont-hot来表示，假设词典D的大小是N，那么词典中位置第i个的词语的表达就是一个$N\times 1$的列向量，该向量中除了第i个位置是1，其他元素皆为0。**那么这就化为了多分类问题，分类数就是N。**    
其网络图是  
![pic1](../pic/Selection_027.jpg)  
其中$\bold v(context(w)_1)$就是w前面n个词中第一个词的词向量，该词向量的维度是$R^{m\times 1}$。我们说用one-hot来表示词语，为什么输入突然变成了这样一个词向量呢？其实很简单，主要是这个图省略了一步从原始输入（one-hot表示）到input的这个层，那这个层具体怎么构建？ 将w的前n个词的one-hot表示的向量转置后一行行放到一个矩阵G中，G的维度就是$R^{n\times N}$，原始输入到input层的权重矩阵$W_o$的维度为$R^{N\times m}$，那么input层$V=GW_o$维度就是$R^{n\times m}$，就是前面说的w前面n个词的词向量。其中要注意的是，$W_o$的第i行就是第i个词的词向量，因为每个词都用one-hot表示，所以第i个次的one-hot表示的h_i和$W_o$相乘后，正好把$W_o$的第i行给提取出来了。  
投影层就是将这n个词向量转置后一行行的放到一个矩阵$X_w$中。然后就是常规的神经网络的向前传递：  
$$
Z_w = \tanh(WX_w+p) \\
y_w = UZ_w+q
$$
$y_w$维度是$R^{N\times 1}$，即使对词典中每一个词的score。通过softmax转换为概率，在通过交叉熵构造损失函数（和以前一样，词w的真实概率可以用one-hot来代替），然后通过梯度下降就可以进行训练了。关键是**由于词典N很大，造成在输出层$y_w$和softmax计算量非常大，严重制约了训练速度！**  

## Word2vec 
word2vec是建立在Benjio算法的基础上，通过一些方法来减少计算量，从而能够很快的得到词向量。具体模型有两个：CBOW model 和 skip-gram model。在减少计算量上，方法有两种：hierarchical softmax 和 negative sampling。另外为了减少计算量，还将隐藏层给拿掉了。  
![pic2](../pic/Selection_028.jpg)
看CBOW模型，同样，输入层$context(w)$词向量的由来也省略了原始层和输入层的映射，其权重矩阵的每一行就是对应单词的词向量。input到projection，是将$context(w)$词向量求和，projection到output没有用Benjio的全连接，直接就给出了词w的预测，这当然是为了减少计算量，但这是怎么做到的？这就用到了hierarchical softmax和negative sampling。

### CBOW + Hierarchical softmax

![pic3](../pic/Selection_029.jpg)
从图中可以看到，projection 到 output 构造了一棵Huffman树。huffman树也叫最小加权路径算法。与堆的结构相似，但构造过程不同。找出词典中每一个词的词频，以词频作为权重，每个词作为节点，构造huffuman树。例如：
![pic4](../pic/Selection_030.jpg)
以词频作为权重构造Huffman树是有原因的，明白如何从Huffman树中求得$p(w|context(w))$，就知道为何使用词频为权重了。  
Hierarchical softmax 就是依靠Huffman树将原来应该是对N个分类应用softmax的问题，转化为分层的Logistics问题。具体做法：  
就是给每个节点编码，左子树编码为1，右子树编码为0，这就是Huffman编码。可以将编码为1的认为是正类，编码为0的认为是负类。以上图中“巴西”为例子，令w='巴西'。如何预测到w呢？从根节点开始，先预测到权重为23节点正类，然后预测到权重为9的负类，然后在预测到正类就预测出w的概率了。对同一个$X_w$使用Logistics回归预测过程依次如下：
$$
p(node_{23}) = \sigma ((\theta_1^w)^TX_w) \\
p(node_{9}) = 1-\sigma((\theta_2^w)^TX_w) \\
p(w) = \sigma((\theta_3^w)^TX_w)
$$
那么$p(w|context(w))$如何求：
$$
p(w|context(w))=p(node_{23})p(node_9)p(w)
$$
可以看出如果w离根节点越远，那么计算$p(w|context(w))$的步骤就越多，不论w周围的词怎么样，要计算$p(w|context(w))$所需要的步数是一样的，假设计算步数为k(上面'巴西'概率计算需要3步，k=3)，w出现的频率为$n_w$，那么总的计算步数就为$kn_w$，为了减少计算量，我们将词频出现越高的，那么就离根节点越近，这就是用用词频构造Huffman的原因。  
计算出$p(w|context(w))$之后，在用交叉熵就可以构造该窗口下的损失函数了，将所有窗口的损失函数求和就是总的损失函数，然后用梯度下降法就可以进行训练了。  

### skip-gram + Hierarchical softmax
有了CBOW的铺垫，该模型就很好理解了。
这里有一个条件独立假设：
$$
p(context(w)|w) = \prod_{u\in context(w)} p(u|w)
$$
而每一个$p(u|w)$的求法和上面用 Hierarchical softmax是一样的。

### CBOW + Negative sampling
CBOW 模型给定$context(w)$预测w，那么w就是正样本，不是w的就是负样本。我们依照一定的采样方法，采样出一定个数的负样本，就叫负采样。
那么$p(w|context(w))$怎么求呢？    
对与投影层的$X_w$，可以给一个权重向量$\theta_w$，从而得到分类为w的score，如果在给一个权重向量$\theta_u$，得到分类为u词时的score，对字典中的每一个词赋予一个权重向量，就得到了分类关于该词的score，这就右落入了投影层到输出层是全连接，然后用softmax问题的原点了。  
现在对与投影层的$X_w$，给一个权重向量$\theta_w$，得到分类为w的score，现在不是取字典中的所有词，而是取负采样的得到的词，对每一个词给一个权重向量$\theta_u$，得到负采样中每个词的score，接下来是对集合$W_s=w\cup Neg(w)$使用softmax么？似乎可以，但作者没有这么做，可能是因为负采样过多，计算量也会很大的缘故（自己猜测）。  
~~将$W_s$看做是一个二分类的训练集，然后对这个训练集采用最大似然函数作为目标函数（完全当做二分类问题来做，集合$W_s$中除了w样本为1，其他皆为0)~~ 此处理解有误。
正确的理解是：**将每一个预测结果当成二分类，当预测是w时，其他不是w的都是负类，那么预测w的概率就是$\sigma(\theta_w^TX_w)$，预测负样本中的词u时，将其他不是u的词当做负类，那么u的概率就是$\sigma(\theta_u^TX_w)$**，现在我们想要最大化预测w词的概率，同时最小预测负采样中词的概率(反过来想，就是最大化$1-\sigma(\theta_u^TX_w))$，那么很自然想到构造的目标函数就是:
$$
f(W_s)  = \sigma(\theta_w^TX_w) \prod_{u \in Neg(w)}(1-\sigma(\theta_u^TX_w))
$$
对其取负对数就是该窗口的损失函数了，将所有窗口的损失函数加起来，就是总的损失函数，然后用梯度下降法（向后传播）就可以训练了。

### skip-gram + Negative sampling
由中心词w预测周围的词$context(w)$，对每一个要预测的词$a\in context(w)$，都对其进行负采样得到$Neg(a)$，然后得到$f(a\cup Neg(a))$，继而得到目标函数：
$$
F = \prod_{a\in context(w)} f(a\cup Neg(a))
$$
接下来的操作步骤就一样了。

### Negative sampling
上面没有说明是如何进行负采样的，摘录《word2vec中的数学原理详解》，本文大部分都来自这篇文章，并加入了自己的理解，其中关于负采样的部分，《详解》中的解释不对，但是公式是对的。
![pic5](../pic/Selection_032.jpg)
![pic6](../pic/Selection_033.jpg)
