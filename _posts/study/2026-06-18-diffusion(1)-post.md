---
title: 扩散模型理解
date: 2026-06-18 00:00:00 +0800
categories: [study]
tags: [diffusion, diffusion model, diffusion process]
pin: true
math: true
mermaid: true
---

## Diffusion 公式基础讲解
### 1. Diffusion 过程
Diffusion 过程可以看作是高斯噪声逐渐添加到图像上的过程。假设我们有一个图像 $x$，我们希望将其添加噪声，使其变为 $x_t$。这个过程可以看作是一个马尔可夫过程，即 $x_t$ 只依赖于 $x_{t-1}$。

我们可以将这个过程表示为：

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t
$$

其中，$\epsilon_t \sim \mathcal{N}(0, I)$. $\alpha$ 和 $\beta$ 是超参数，$\epsilon_t$ 是高斯噪声。

基于此公式，我们可以直接从$x_0$得到$x_t$，下面给出推导。

在进行推导之前，我们需要先理解一个概念，即两个独立高斯分布的和仍然是高斯分布，其均值和方差分别为两个高斯分布的均值和方差的和。公式化表示如下：

$$
p \sim N(\mu_1, \sigma_1^2), \quad q \sim N(\mu_2, \sigma_2^2)
$$

则有 $p+q \sim N(u_1+u_2, \sigma_1^2+\sigma_2^2)$，写成具体采样过程如下：

$$
p = \mu_1 + \sigma_1 \epsilon_1, \quad q = \mu_2 + \sigma_2 \epsilon_2
$$

其中，$\epsilon_1 \sim \mathcal{N}(0, I), \epsilon_2 \sim \mathcal{N}(0, I)$。
则 $p+q = \mu_1 + \sigma_1 \epsilon_1 + \mu_2 + \sigma_2 \epsilon_2 = (\mu_1 + \mu_2) + \sqrt{\sigma_1^2 + \sigma_2^2} \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

其中，最关键的公式为：

$$
\begin{equation}
\sigma_1 \epsilon_1 + \sigma_2 \epsilon_2 = \sqrt{\sigma_1^2 + \sigma_2^2} \epsilon
\label{eq:1}
\end{equation}
$$

即两个高斯分布的和的方差等于两个高斯分布的方差的和。**这是后面可以一步加噪的关键！**

#### 1.1 从 $x_0$ 到 $x_t$

$$
\begin{equation}
\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t \\
&= \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-1}) + \sqrt{1 - \alpha_t} \epsilon_t \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \epsilon_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t \\
&= \sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}} x_{t-3} + \sqrt{\alpha_t \alpha_{t-1} (1 - \alpha_{t-2})} \epsilon_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \epsilon_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t \\
&= \cdots \\
&= \prod_{i=1}^{t} \sqrt{\alpha_i} x_0 + \underbrace{\sum_{i=1}^{t} \sqrt{\alpha_t\cdots\alpha_{t-i+1}-\alpha_t\cdots\alpha_{t-i}} \epsilon_i}_{\mathcal{A}} 
\end{aligned}
\end{equation}
$$

这里的$\mathcal{A}$部分，即噪声项，可以使用Eq\eqref{eq:1}进一步化简：

$$
\sum_{i=1}^{t} \sqrt{\alpha_t\cdots\alpha_{t-i+1}-\alpha_t\cdots\alpha_{t-i}} \epsilon_i = \sqrt{1 - \prod_{i=1}^{t} \alpha_i} \epsilon
$$

如果令$\bar{\alpha_t} = \prod_{i=1}^{t} \alpha_i$，则最终的公式为：

$$
x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1 - \bar{\alpha_t}} \epsilon
$$

如果是从分布角度来看，则可以看作是 $x_0$ 经过 $t$ 步的扩散过程，最终得到 $x_t$。
从$x_{t-1}$ 到 $x_t$ 的过程，可以看作是 $x_{t-1}$ 经过一步的扩散过程：

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t)
$$

#### 1.2 从 $x_t$ 到 $x_{t-1}$
![alt text](/assets/img/papers/study/image.png)
反过来：

$$
\begin{equation}
\begin{aligned}
q(x_{t-1}|x_t) &= \frac{q(x_t, x_t, x_0)}{q(x_t, x_0)} \\
&= \frac{q(x_t|x_{t-1})q(x_{t-1}|x_0)q(x_0)}{q(x_t|x_0)q(x_0)} \\
&= \frac{q(x_t|x_{t-1})q(x_{t-1}|x_0)}{q(x_t|x_0)}
\end{aligned}
\end{equation}
$$

由1.1节可知：

$$
\begin{aligned}
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, 1 - \alpha_t) \\
q(x_{t-1}|x_0) &= \mathcal{N}(x_{t-1}; \sqrt{\alpha_{t-1}} x_0, 1 - \alpha_{t-1}) \\
q(x_t|x_0) &= \mathcal{N}(x_t; \sqrt{\bar{\alpha_t}} x_0, 1 - \bar{\alpha_t})
\end{aligned}
$$

因此：

$$
q(x_{t-1}|x_t, x_0) = \mathcal{N} (x_{t-1}; \frac{\sqrt{\alpha_t}(1-\bar{\alpha_{t-1}})x_t + \sqrt{\bar{\alpha_{t-1}}}(1-\alpha_t)x_0}{1-\bar{\alpha_t}}, \frac{(1 - \alpha_t)(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_t}})
$$

实际上，在反向过程中，我们是不知道 $x_0$ 的。
但我们可以根据1.1中的公式，推导出$x_0$:

$$
x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1-\bar{\alpha}}\epsilon_t \\
\Rightarrow x_0 = \frac{x_t - \sqrt{1-\bar{\alpha_t}}\epsilon_t}{\sqrt{\bar{\alpha_t}}} \\
\Rightarrow \frac{\sqrt{\alpha_t(1-\bar{\alpha_{t-1}})}x_t + \sqrt{\bar{\alpha_{t-1}}}(1-\alpha_t)x_0}{1-\bar{\alpha_t}} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}}\epsilon_t)
$$

这里，我们只需要学习$\epsilon_t$即可完成整个推理过程。
