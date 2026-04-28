---
title: microgpt 用 200 行代码揭秘神经网络训练算法
description: microgpt.py 是 Andrej Karpathy 最近刚刚发布的一个 200 行 python 代码文件，极简地指出了 GPT 的训练与推理算法。
date: 2026-04-10 18:30:00 +0800
categories: [AI, Training]
tags: [thinking]
---

*microgpt.py 是 Andrej Karpathy 最近刚刚发布的一个 200 行 python 代码文件，极简地指出了 GPT 的训练与推理算法。这篇文章是对 microgpt 的详细解析。*

今天谈大语言模型，人很容易先被“巨大”这个事实震住。几千亿参数，海量语料，分布式训练，GPU 集群，复杂框架。于是我们往往会形成一种错觉：神经网络太复杂、太难懂了。

但**真正让人看不清的，不是规模，而是封装。**数据怎样变成输入，输入怎样变成预测，预测怎样变成损失，损失又怎样沿着计算图传回参数，最后参数怎样真的被改掉——这条链在工业系统里通常被包得太好。你能调用它，能运行它，能得到结果，却未必真正看见它。

microgpt 的珍贵，恰恰在这里。它不是要训练一个强大的模型，而是要把最核心的训练链条尽可能讲明白。**它仅用 200 行 python 代码，把字符级 tokenizer、前向传播、损失计算、自动微分、Adam 更新和自回归采样整个过程彻底展现出来。**

大模型最深的秘密，并不只在参数规模里，它更在这条循环里。microgpt 的意义，就是把这条循环压缩到你终于可以一眼看清的程度。

### （一）模型最先学到的，不是意义，而是分布

microgpt 使用了一个简单的数据集。它读取一个名字数据集，每一行是一条字符串；如果本地没有文件，可以去下载 Karpathy 代码中的 `names.txt`。接着，它把所有出现过的字符收集出来，为每个字符分配一个整数 id，并额外增加一个 `BOS` 作为序列起点。

```python
# 下载并读取数据集
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]

# 构建字符级词表
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
```

于是，一条训练样本被编码成 `[BOS] + 字符序列 + [BOS]`。

```python
    # 把字符串转成 token 序列，并在首尾都加上 BOS
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

这个设计同时规定了两件事：从哪里开始，以及在哪里结束。

这一步看起来基础，却已经决定了模型究竟在学什么。**它学的不是“这个名字意味着什么”，而是更底层的东西：在给定前文的情况下，下一个字符更可能是什么。换句话说，它学的是概率分布。**

看到起点标记之后，哪些字符更像名字开头；看到某几个字符之后，哪些延续更可能出现；随着前文不断延长，这个分布也不断收缩、重排、更新。**模型不是在背答案，而是在逐步改写自己对“下一步最可能是什么”的估计。**

这也是为什么到了后面的推理阶段，一旦再次采样到 `BOS`，生成就会结束。训练阶段学到的，不只是如何展开一个序列，也包括如何把一个序列收束回来。

### （二）前向传播，本质上是在形成判断

平时我们常说“模型预测下一个 token”。这句话没有错，但太快了。**它把中间最重要的过程压缩掉了：模型到底凭什么做出这个判断。**

在 microgpt 里，这个过程被拆得非常清楚。取出当前 token 对应的词嵌入向量 `tok_emb`，再取出当前位置对应的位置嵌入向量 `pos_emb`，然后相加得到当前位置的输入表示。离散 token id 本身只是编号，不能直接承载可学习的结构；**只有进入连续向量空间，模型才有可能通过点积、加权、归一化这些操作，把上下文压成一种可计算的表示。**位置向量同样关键，它把“是什么”与“在什么位置”合在了一起。接着，**模型对这个表示做归一化，这里使用的是 RMSNorm(Root Mean Square Layer Normalization，均方根层归一化) 。**

```python
def gpt(token_id, pos_id, keys, values):
    # 取出当前 token 对应的词嵌入向量
    tok_emb = state_dict['wte'][token_id] # token embedding
    # 取出当前位置对应的位置嵌入向量
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    # 把词嵌入和位置嵌入逐元素相加，得到输入表示
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    # 对输入表示做一次 RMSNorm
    x = rmsnorm(x)
```

接着，进入注意力模块。这里会生成 q、k、v 三组向量。最直白地说，**q 是当前位置提出的问题，k 是历史位置留下的索引，v 是那些位置真正携带的内容。当前位置会拿自己的 query 去和历史 key 打分，分数经过 softmax 变成权重，再用这些权重对历史 value 加权求和。**这个结构之所以深刻，不在于术语本身，而在于它把两件事分开了：先决定“看谁”，再决定“拿什么”。

如果说 attention 负责让当前位置吸收历史上下文，那么后面的 MLP （Multi-Layer Perceptron）多层感知机更像是在当前位置内部继续加工这些信息。模型先扩张维度，再经过非线性，再压回原维度，并通过残差把旧表示保留下来。

```python
        # 1) 多头自注意力模块
        # ...
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        # ... (计算注意力权重并对v加权求和得到 x_attn) ...
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # 2) 前馈 MLP 模块
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
```

于是，**一个完整的 Transformer block 可以被理解成两次连续改写：先向历史发问，再在当前位置内部重组答案。**全部这些加工完成之后，模型还没有直接输出“答案”，它只是把当前位置的隐藏状态映射到整个词表，形成一组 `logits`。

```python
    # 最后把当前隐藏状态映射到词表大小维度，得到每个 token 的 logits
    logits = linear(x, state_dict['lm_head'])
    return logits
```

也就是说，前向传播的终点，不是结论，而是一张分数地形。**真正的答案，还要等 softmax 把它变成分布之后才出现。**

### （三）真正让模型会学的，是误差如何回传到参数

到这里为止，模型已经形成了判断，但还没有真正学会。因为判断本身不会自动改写模型。**只有当这个判断和真实答案发生比较，并且把差异转成一种可以回传的信息时，训练才真正开始。**

microgpt 最有教育意义的地方，就在这里。它没有把这件事交给现成框架，而是自己实现了一个极简的 Value 类。`Value` 不是普通数字，而是带着来路和梯度的标量节点。每个节点都保存当前数值、它依赖的子节点，以及它对每个子节点的局部导数。

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') 

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 保存标量值
        self.grad = 0                   # 保存梯度
        self._children = children       # 依赖的子节点
        self._local_grads = local_grads # 对各个子节点的局部导数
```

如此一来，**模型里的每一步运算都不会“算完就消失”，而会把“我是怎样得到的”一并记录下来。**

这件事的本质，是把链式法则预埋进计算图。一次乘法，不再只是得到一个数，而是得到一个知道自己来自哪里、也知道将来该怎样把误差信号传回去的节点。于是，从参数到 logits、从 logits 到概率、从概率到损失，整条路径都会被串成一张完整的图。

**训练时，代码在序列的每个位置上都做一次 next-token prediction：输入当前位置的 token，目标是下一个位置的 token，得到 logits，经 softmax 变成概率，再取正确 token 的负对数作为该位置的损失，最后对整条序列求平均。于是，一个样本的整条误差链就完整地搭起来了。**

```python
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
        
    loss = (1 / n) * sum(losses)
    loss.backward()
```

接下来发生的，就是 `loss.backward()`。它先对整张图做拓扑排序，再按逆拓扑序把梯度一层层传回去。代码里那句 `child.grad += local_grad * v.grad`，几乎就是反向传播全部秘密的浓缩：当前节点对损失的敏感度，乘上当前节点对其子节点的局部导数，就变成了子节点该承担的责任。

```python
    def backward(self):
        # ... (构建拓扑序 topo) ...
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # 链式法则
                child.grad += local_grad * v.grad
```

这里的 `+=` 很重要，因为一个节点可能通过多条路径共同影响最终损失，它的总梯度必须是所有路径贡献之和。

**反向传播并不是把一个错误简单地往前扔，而是在整张图上做一次责任汇总与责任分配。**

### （四）梯度只是方向，更新才是改变

模型拿到梯度之后，还没有真正完成学习。梯度告诉它：哪些参数负有责任，该往哪个方向移动。但**“知道方向”还不等于“真正改掉”。这中间还隔着优化器。**

microgpt 用的是一种叫做 Adam 的优化器。它不是只看当前这一步的梯度，而是为每个参数维护两类历史统计量：一类是一阶矩，也就是方向上的滑动平均；另一类是二阶矩，也就是尺度上的滑动平均。**这样做的意义在于，参数更新不再只是听从当前梯度的一次指令，而是会结合近期趋势与波动幅度，做出更稳妥的移动。某些方向如果持续一致，优化器会更有信心；某些梯度如果波动过大，优化器会更谨慎。**

与此同时，代码还让学习率随训练步数线性下降。训练前期，模型还在大范围搜索，步子可以大一些；训练后期，模型逐渐接近更稳定的区域，更新就应该更细、更慢。

最后是 `p.grad = 0`。这行代码看起来微不足道，实际上很关键。

```python
    lr_t = learning_rate * (1 - step / num_steps) # 线性衰减
    for i, p in enumerate(params):
        # 更新一阶矩与二阶矩估计
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # 偏差校正
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        
        # 按 Adam 公式更新参数值
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        # 参数更新完后把梯度清零
        p.grad = 0
```

因为在这个系统里，梯度是累加到参数对象上的。**如果不清零，上一轮的责任就会被错误地带进下一轮。清零不是把知识抹掉，而是让这一次责任分配正式结案，好让下一次判断重新开始。**

至此，一个完整训练步的闭环才真正成立：前向给出判断，损失产生误差，backward 追究责任，Adam 改写参数，梯度清零，准备下一轮。

### （五）生成不是吐出被记住的答案，而是沿着概率分布重新展开

如果文章停在优化器，其实还差最后一层。因为训练最终不是为了把损失降下来，而是为了让模型在推理时真的能生成东西。

microgpt 的推理部分极其干净。它从 BOS 开始，调用同一个 `gpt(token_id, pos_id, keys, values)` 做前向传播，得到 logits，经 softmax 变成概率分布，再按概率采样一个新的 token。如果采到 BOS，就结束；否则把新 token 接回去，进入下一步。

```python
temperature = 0.5 # 控制采样温度

for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        # 缩放 logits 并转成概率分布
        probs = softmax([l / temperature for l in logits])
        # 按概率随机采样
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
```

也就是说，**推理阶段并没有跳出训练阶段建立起来的机制，它只是把“正确答案 + 损失”换成了“采样 + 继续”。**训练是在不断改写分布，推理则是在这套分布上一步步往前走。

这也解释了 temperature 的作用。**采样前，代码会先用 temperature 缩放 logits，再做 softmax。温度低，分布更尖锐，高概率 token 更容易被选中，生成更稳；温度高，分布更平，低概率 token 更容易冒出来，生成也更发散。**

关键不在于“随机性是不是更好”，而在于这再次提醒我们：模型在任何时刻维护的首先都是一张分布，而不是一个现成答案。所谓生成风格的变化，实际上只是这张分布形状的变化。这里设置的 `temperature = 0.5`，对应的正是一种更偏稳定的采样。

另一个容易被忽略的点是，**推理阶段同样需要 keys 和 values 缓存。**原因很简单：注意力机制并没有变。当前位置仍然要“看见过去”，才能决定下一步往哪里走。所以 KV cache 不是训练专属的技巧，它既服务训练，也服务生成。

训练和推理共享同一套前向逻辑，只是在尾部接上了不同的东西：一个接损失，一个接采样。

### 小结

到这里，整个 microgpt 逻辑才真正闭合起来。

**它从一个极简字符级数据集出发，把输入编码成 token；再把 token 放进 embedding 和 attention 所构成的表示系统；用前向传播形成对下一个 token 的判断；用损失和 backward 把误差变成可传播的梯度；再用 Adam 把梯度落实成参数更新；最后从 BOS 出发，让已经被改写过的参数在推理中一步步展开成新序列。**它不是高性能实现，却几乎以最透明的方式，把 GPT 的核心算法链路全部暴露出来了。

它让通常被规模和工程噪声遮住的训练原理重新显露出来。大模型最深的秘密，不在于参数规模本身，而在于它把“预测—比较—追责—更新—再预测”这条闭环做到了极致。

microgpt 的价值，就是把这条闭环缩回到了人脑仍然可以直接思考的尺度。

**神经网络训练的本质，不是让模型记住答案，而是让它在一次次预测错误之后，把误差信号沿计算图传回参数，并逐步改写自己对下一个 token 的概率分布。**


*参考资料：https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95*
