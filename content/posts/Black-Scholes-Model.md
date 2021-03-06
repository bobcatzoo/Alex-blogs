---
title: "Black-Scholes-Merton Model"
date: 2021-06-11 23:48:25+0800
draft: false
tags: ["Finance"]
categories: ["欧式期权定价"]
markup: mmark
---

2008年的1次考证经历，第1次知道了期权并学到了期权定价的Black-Scholes-Merton公式，被其深深迷住，断断续续翻阅了许多书籍及文献资料，包括Hull那本及基础的数学分析、实分析及基于测度论的概率论等等，但由于大学专业和数学不相关，看起来很吃力，虽有收获但感觉却与​Black-Scholes-Merton公式渐行渐远，需要补习的基础知识太多。直到工作后于2015年左右开始读世图影印出版的*Stochastic Calculus for Finance II*方才有种醍醐灌顶的感觉，似懂非懂中看到第5章开头后就因种种原因放弃了。

时间一直来到了2021年，趁*COVID-19*疫情期间工作之余，不忘初心又重新拾起那段记忆，从头开始又来到了第5章，过往的疑惑及不解在此刻正逐步消散。

在此结合*Stochastic Calculus for Finance II*前5章的学习，对风险中性测度下的Black-Scholes-Merton理论推导做个总结，并结合C++和Python实现其定价公式，以便对给定的参数，计算出期权价格。希望若干年以后还能不忘初心，照着笔记顺藤摸瓜🍉


### 1. 风险中性测度下的股价过程

#### 1.1 概率测度$\mathbb{P}$下的股价过程

假设 $W(t), 0 \leq t \leq T $是概率空间$(\Omega, \mathcal{F}, \mathbb{P})$上的布朗运动，$\mathcal{F}(t), 0\leq t \leq T$是该布朗运动的域流，考虑股价$S(t)$，其微分如下：

$$
d S(t)=\alpha(t) S(t) d t+\sigma(t) S(t) d W(t), \quad 0 \leq t \leq T.  \tag{1.1}
$$

其中平均回报率$\alpha(t)$和股价波动率$\sigma(t)$为适应性过程，则$S(t)$满足以下等式：

$$
S(t)=S(0) \exp \left\{\int_{0}^{t} \sigma(s) d W(s)+\int_{0}^{t}\left[\alpha(s)-\frac{1}{2} \sigma^{2}(s)\right] d s\right\} \tag{1.2}
$$

假设我们有适应性利率过程$R(t)$，定义折现过程

$$
D(t)=e^{-\int_{0}^{t} R(s) d s} \tag{1.3}
$$

则

$$
d D(t)=-R(t) D(t) d t \tag{1.4}
$$

$D(t)S(t)$及其微分$d (D(t)S(t))$分别为：

$$
D(t) S(t)=S(0) \exp \left\{\int_{0}^{t} \sigma(s) d W(s)+\int_{0}^{t}\left[\alpha(s)-R(s)-\frac{1}{2} \sigma^{2}(s)\right] d s\right\} \tag{1.5}
$$

$$
\begin{aligned}
d(D(t) S(t)) &=[\alpha(t)-R(t)] D(t) S(t) d t+\sigma(t) D(t) S(t) d W(t) \\
\\
&=\sigma(t) D(t) S(t)[\Theta(t) d t+d W(t)]
\end{aligned} \tag{1.6}
$$

其中定义风险的市场价格$\Theta(t)=\frac{\alpha(t)-R(t)}{\sigma(t)} $。

#### 1.2 *Girsanov Theorem*

假设$W(t), 0 < t < T$是概率空间$(\Omega, \mathcal{F}, \mathbb{P})$上的布朗运动，$\mathcal{F}(t)$为该布朗运动的域流，$\Theta(t), 0 < t < T$为适应性过程。我们定义

$$
Z(t) =\exp \left\{-\int_{0}^{t} \Theta(u) d W(u)-\frac{1}{2} \int_{0}^{t} \Theta^{2}(u) d u\right\} \tag{1.7}
$$

$$
\widetilde{W}(t) =W(t)+\int_{0}^{t} \Theta(u) d u \tag{1.8}
$$

并假设

$$
\mathbb{E} \int_{0}^{T} \Theta^{2}(u) Z^{2}(u) d u<\infty \tag{1.9}
$$

让$Z = Z(T)$。那么$\mathbb{E}Z = 1$且在由

$$
\widetilde{\mathbb{P}}(A) = \int_{A}Z(\omega) d \mathbb{P}(\omega) \quad \text{对所有} A \in \mathcal{F}. \tag{1.10}
$$

定义的概率测度$\widetilde{\mathbb{P}}$下，$\widetilde{W}(t), 0 < t < T$是布朗运动。$\Box$

#### 1.3 概率测度$\widetilde{\mathbb{P}}$下的股价过程

根据*Girsanov Theorem*，在概率测度$\widetilde{\mathbb{P}}$下，$d \widetilde W(t) = \Theta(t) d t+d W(t) $，因此公式$(1.6)$也可以写为

$$
d (D(t)S(t)) = \sigma(t) D(t) S(t) d \widetilde W(t) \tag{1.11}
$$

两边同时积分

$$
D(t) S(t)=S(0)+\int_{0}^{t} \sigma(u) D(u) S(u) d \widetilde{W}(u) \tag{1.12}
$$

由于在概率测度$\widetilde{\mathbb{P}}$下，$\int_{0}^{t} \sigma(u) D(u) S(u) d  \widetilde{W}(u)$是伊藤过程，因此是一个鞅。

**因此我们称*Girsanov Theorem*下的概率测度$\widetilde{\mathbb{P}}$为风险中性测度(risk-neutral measure)**。

将$d \widetilde W(t) = \Theta(t) d t+d W(t) $带入公式$(1.1)$，可以得到在概率测度$\widetilde{\mathbb{P}}$下，公式$(1.1)$和$(1.2)$分别可以改写成公式$(1.13)$和公式$(1.14)$的形式

$$
d S(t)= R(t) S(t) d t+\sigma(t) S(t) d \widetilde {W}(t) \tag{1.13}
$$

$$
S(t) = S(0) \exp \left\{\int_{0}^{t} \sigma(s) d \widetilde{W}(s)+\int_{0}^{t}\left[R(s)-\frac{1}{2} \sigma^{2}(s)\right] d s\right\} \tag{1.14}
$$

### 2. 风险中性测度下投资组合的价值过程

假设投资者初始资本为$X(0)$，在任意时间$t, 0 < t < T $持有$\Delta(t)$份股票，同时以利率$R(t)$投资或借贷于货币市场，以维持自融资状态，则投资组合价值的微分为

$$
\begin{aligned}
d X(t) &=\Delta(t) d S(t)+R(t) \left[ X(t)-\Delta(t) S(t) \right] d t \\
\\
&=\Delta(t) [\alpha(t) S(t) d t+\sigma(t) S(t) d W(t)]+R(t)[X(t)-\Delta(t) S(t)] d t \\
\\
&=R(t) X(t) d t+\Delta(t)[\alpha(t)-R(t)] S(t) d t+\Delta(t) \sigma(t) S(t) d W(t) \\
\\
&=R(t) X(t) d t+\Delta(t) \sigma(t) S(t)[\Theta(t) d t+d W(t)]
\end{aligned} \tag{2.1}
$$

根据$\text{Ito}$乘法法则，由公式$(1.4)$和$(1.6)$可得

$$
\begin{aligned}
d(D(t) X(t)) &=\Delta(t) \sigma(t) D(t) S(t)[\Theta(t) d t+d W(t)] \\
\\
&=\Delta(t) d(D(t) S(t)) \\
\\
&=\Delta(t) \sigma(t) D(t) S(t) d \widetilde W(t)
\end{aligned} \tag{2.2}
$$

由此投资者有两种选择：

1. 以利率$R(t)$投资于货币市场；

2. 在风险中性测度$\widetilde{\mathbb{P}}$下投资于平均回报率为$R(t)$的股票。

但由于在风险中性测度$\widetilde{\mathbb{P}}$下，投资组合的折现价值$D(t)X(t)$是鞅，因此不管投资者如何选择，其投资组合的平均回报率均为为$R(t)$。

### 3. 风险中性测度下的期权定价

我们令$\mathcal{F}(T)$可测的随机变量$V(T)$表示在时刻$T$衍生证券空头的潜在支付(*payoff*) $(S(T)-K)^+$，投资者为了对冲看涨期权空头即未来所面临的潜在支出$V(T)$，那么其持有的投资组合$X(t)$需要使以下等式几乎必然成立(*almost surely*)

$$
X(T) = V(T) \tag{3.1}
$$

我们先假设公式$(3.1)$成立，并由此确定初始资本$X(0)$和$\Delta t$过程。由$D(t)X(t)$在测度$\widetilde{\mathbb{P}}$是鞅的事实我们有

$$
D(t) X(t)=\widetilde{\mathbb{E}}[D(T) X(T) \mid \mathcal{F}(t)]=\widetilde{\mathbb{E}}[D(T) V(T) \mid \mathcal{F}(t)] \tag{3.2}
$$

$X(t)$表示在时刻$t$为完全对冲衍生证券支付$V(T)$所持有的投资组合价值，我们将其称之为衍生证券在时刻 $t$ 的价格并用$V(t)$表示，那么公式$(3.2)$可以写成

$$
D(t) V(t)=\widetilde{\mathbb{E}}[D(T) V(T) \mid \mathcal{F}(t)], 0 \leq t \leq T \tag{3.3}
$$

由于$D(t)$是$\mathcal{F}(t)$可测的，因此我们可以将其移到公式右侧，得到

$$
V(t)=\widetilde{\mathbb{E}}\left[e^{-\int_{t}^{T} R(u) d u} V(T) \mid \mathcal{F}(t)\right], 0 \leq t \leq T \tag{3.4}
$$

我们将公式$(3.3)$和$(3.4)$成为连续时间下风险中性定价公式(*risk-neutral pricing formula*)。

### 4. 推导$\text{Black-Scholes-Merton}$公式

为简单起见，我们假设$\sigma(t)$和$R(t)$分别为常数 $\sigma$ 和 $r$，则公式$(3.4)$简化为

$$
\widetilde{\mathbb{E}}\left[e^{-r(T-t)}(S(T)-K)^{+} \mid \mathcal{F}(t)\right] \tag{4.1}
$$

公式$(4.1)$仅依赖于时刻$t$和股价$S(t)$，由于几何布朗运动是马尔可夫过程，因此存在$c(t, S(t))$满足

$$
c(t, S(t)) = \widetilde{\mathbb{E}}\left[e^{-r(T-t)}(S(T)-K)^{+} \mid \mathcal{F}(t)\right] \tag{4.2}
$$

公式$(1.10)$简化为

$$
S(t) = S(0) \exp \left\{\sigma \widetilde{W}(t)+\left(r-\frac{1}{2} \sigma^{2}\right) t\right\} \tag{4.3}
$$

则$S(T)$等于

$$
\begin{aligned}
S(T) &=S(t) \exp \left\{\sigma(\widetilde{W}(T)-\widetilde{W}(t))+\left(r-\frac{1}{2} \sigma^{2}\right) \tau\right\} \\
\\
&=S(t) \exp \left\{-\sigma \sqrt{\tau} Y+\left(r-\frac{1}{2} \sigma^{2}\right) \tau\right\}
\end{aligned} \tag{4.4}
$$

其中$\tau = T - t$，$Y$是标准正态随机变量

$$
Y = - \frac{\widetilde{W}(T) - \widetilde{W}(t)} {\sqrt{T-t}} \tag{4.5}
$$

公式$(4.2)$可以写成如下形式

$$
\begin{aligned}
c(t, x) &= \widetilde{\mathbb{E}}\left[e^{-r \tau}\left(x \exp \left\{-\sigma \sqrt{\tau} Y+\left(r-\frac{1}{2} \sigma^{2}\right) \tau\right\}-K\right)^{+}\right] \\
\\
&= \frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{\infty} e^{-r \tau}\left[ x \exp \left\{-\sigma \sqrt{\tau} y+\left(r-\frac{1}{2} \sigma^{2}\right) \tau\right\}-K\right]^{+} e^{-\frac{1}{2} y^{2}} d y
\end{aligned} \tag{4.6}
$$

其中被积函数

$$
\left[ x \exp \left\{-\sigma \sqrt{\tau} y+\left(r-\frac{1}{2} \sigma^{2}\right) \tau\right\}-K\right]^{+} \tag{4.7}
$$

为正，当且仅当

$$
y < d_{-}(\tau, x)=\frac{1}{\sigma \sqrt{\tau}}\left[\log \frac{x}{K}+\left(r-\frac{1}{2} \sigma^{2}\right) \tau\right] \tag{4.8}
$$

由此

$$
\begin{aligned}
c(t, x) &=\frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{d_{-}(\tau, x)} e^{-r \tau}\left(x \exp \left\{-\sigma \sqrt{\tau} y+\left(r-\frac{1}{2} \sigma^{2}\right) \tau\right\}-K\right) e^{-\frac{1}{2} y^{2}} d y \\
\\
&=\frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{d_{-}(\tau, x)} x \exp \left\{-\frac{y^{2}}{2}-\sigma \sqrt{\tau} y-\frac{\sigma^{2} \tau}{2}\right\} d y \\
\\
&- \frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{d_{-}(\tau, x)} e^{-r \tau} K e^{-\frac{1}{2} y^{2}} d y \\
\\
&=\frac{x}{\sqrt{2 \pi}} \int_{-\infty}^{d_{-}(\tau, x)} \exp \left\{-\frac{1}{2}(y+\sigma \sqrt{\tau})^{2}\right\} d y-e^{-r \tau} K N\left(d_{-}(\tau, x)\right) \\
\\
&=\frac{x}{\sqrt{2 \pi}} \int_{-\infty}^{d_{-}(\tau, x)+\sigma \sqrt{\tau}} \exp \left\{-\frac{z^{2}}{2}\right\} d z-e^{-r \tau} K N\left(d_{-}(\tau, x)\right) \\
\\
&=x N\left(d_{+}(\tau, x)\right)-e^{-r \tau} K N\left(d_{-}(\tau, x)\right)
\end{aligned} \tag{4.9}
$$

其中

$$
d_+(\tau, x) = d_-(\tau, x) + \sigma \sqrt{\tau} = d_{-}(\tau, x)=\frac{1}{\sigma \sqrt{\tau}}\left[\log \frac{x}{K}+\left(r+\frac{1}{2} \sigma^{2}\right) \tau\right] \tag{4.10}
$$

由此我们得到了欧式看涨期权定价公式

$$
\text{BSM}(\tau, x, K, r, \sigma) = x N\left(d_{+}(\tau, x)\right)-e^{-r \tau} K N\left(d_{-}(\tau, x)\right) \tag{4.11}
$$

### 5. 欧式期权定价公式

根据上述对$\text{Black-Scholes-Merton}$期权定价公式的推导，对于当前股价为$S_0$，行权价格为$K$，行权期为$T$，无风险利率为常数$r$，股价波动率为常数$\sigma$的看涨期权，其期权费$c(S_0, K, T, r, \sigma)$为：

$$
c(S_0, K, T, r, \sigma)=N\left(d_{1}\right) S_{0}-N\left(d_{2}\right) K \exp (-r T) \tag{5.1}
$$

其中

$$
d_{1} = \frac{1}{\sigma \sqrt{T}}\left(\log \left(\frac{S_0}{K}\right)+\left(r+\frac{\sigma^{2}}{2}\right) \sqrt{T}\right) \tag{5.2}
$$

$$
d_{2} = \frac{1}{\sigma \sqrt{T}}\left(\log \left(\frac{S_0}{K}\right)+\left(r-\frac{\sigma^{2}}{2}\right) \sqrt{T}\right) \tag{5.3}
$$

$N(x)$为标准正态分布累积分布函数：

$$
N(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{x} \exp \left(-\frac{t^{2}}{2}\right) d t \tag{5.4}
$$

对应看跌期权费$p(S_0, K, T, r, \sigma)$为:

$$
p(S_0, K, T, r, \sigma)=N\left(-d_{2}\right) K \exp (-r T)-N\left(-d_{1}\right) S_0 \tag{5.5}
$$

### 6. 期权定价的C++实现 

#### 6.1 实现$N(x)$函数 - *From Scratch to Boost Library*

定价公式$(1)$-$(5)$中涉及$N(x)$、$\exp(x)$和$\log(x)$等3个函数，其中$\exp(x)$和$\log(x)$已在标准库`<cmath>`中实现，可以直接使用。因此只剩$N(x)$需要我们在标准库外自己实现或寻求其他库的支持。我们按照如下三种方式分别进行实现：

- 多项式逼近法
- 数值积分法
- 调用`Boost`库

##### 6.1.1 多项式逼近法

如$x > 0$，定义$k = 1/(1 + 0.2316419x)$，则$N(x)$可用如下关于$k$多项式进行逼近
$$
\small 1 − \frac{1}{\sqrt{2 \pi}} \exp(-\frac{x^2} {2}) k(0.319381530 + k(−0.356563782 + k(1.781477937 + k(−1.821255978 + 1.330274429k)))) \tag{6.1}
$$

借助$N(x) + N(-x) = 1$可以求得$N(-x)$即$x < 0$时$N(x)$的值。

C++实现如下：

```cpp
const double Pi = 3.141592653589793;
const double root2Pi = sqrt(2.0 * Pi);

double normcdf_poly(double x)
{
    if (x < 0)
        return 1 - normcdf_ploy(-x);
    double k = 1 / (1 + 0.2316419 * x);
    double poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 
                  + k * (-1.821255978 + 1.330274429 * k))));
    double approx = 1.0 - 1.0 / root2Pi * exp(-0.5 * x * x) * poly;
    return approx;
}
```

##### 6.1.2 数值积分法

对于实数域上$\mathbb R \to \mathbb R$的黎曼积分

$$
F(b) - F(a) = \int_{a}^{b} f(x) d x \tag{6.2}
$$

令$\Delta x = \frac{(b-a)}{N}$，根据黎曼积分定义，我们可以用以下矩形的面积和逼近$\int_{a}^{b} f(x) d x$：

$$
\lim_{N \to \infty} \sum_{i=0}^{N-1} f\left(a+(i+\frac{1}{2}) \Delta x\right) \Delta x \tag{6.3}
$$

只要$N$取值足够大即可很好的逼近积分值，可以通过定义$f(x) = x ^ 2 + 1$进行简单验证，C++实现如下：

```cpp
#include <iostream>
#include <cmath>
using namespace std;

double f_x(double x)
{
    return pow(x, 2) + 1;
}

double integrate_fx(double a, double b, int N)
{
    double delta = (b - a) / N;
    double integrate = 0.0;
    int i = 0;
    while (i < N)
    {
        double x = a + (i + 0.5) * delta;
        integrate += f_x(x) * delta;
        i++;
    }
    return integrate;
}

int main()
{
    double a = 1;
    double b = 3;
    int N = 1000;
    double integrate_ = 0.0;
    
    integrate_ = integrate_fx(a, b, N);
    cout << integrate_ << endl; //返回10+2/3(约等于10.666667)
    
    return 0;
}
```

下面我们考虑如何用数值积分法逼近

$$
N(x)=\frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{x} \exp \left(-\frac{t^{2}}{2}\right) d t
$$

由于积分下限为$- \infty$，我们需要考虑进行变量替换，定义$t = x + 1 - \frac{1} {s}$，则$N(x)$等价于如下积分：

$$
\frac{1} {\sqrt {2 \pi}} \int_{0}^{1} \frac{1}{s^{2}} \exp \left(-\frac{\left(x+1-\frac{1}{s}\right)^{2}}{2}\right) \mathrm{d} s \tag{6.4}
$$

C++实现如下：

```cpp
double normcdf_integrate(double x)
{
    int a = 0;
    int b = 1;
    double N = 1000;
    double delta = 1 / N;
    double integrate = 0.0;
    for (int i = 0; i < N; i++)
    {
        double s = a + (i + 0.5) * delta;
        double t = x + 1 - pow(s, -1);
        double f = pow(2 * PI, -0.5) * pow(s, -2) * exp(-0.5 * t * t);
        integrate += f * delta;
    }
    return integrate;
}
```

##### 6.1.3 调用`boost`库

我们可以通过如下方式调用`boost`库中的累积分布函数（**C**umulative **D**istribution **F**unction，简称$\mathrm{cdf}$)的实现

```cpp
#include "boost\math\distributions\normal.hpp"
#include <random>

double normcdf_boost(double x)
{
	double cdf = 0.0;
	
	boost::math::normal_distribution<> norm(0, 1); //生成标准正态分布
	cdf = boost::math::cdf(norm, x); //计算N(x)值
	return cdf;
}
```

#### 6.2 期权费计算的C++实现

实现$N(x)$后，计算公式$(1)$中欧式看涨期权$c(S_0, K, T, r, \sigma)$的障碍就清除了。下面我们分别定义`blackScholesCallPrice`和`blackScholesPutPrice`函数完成计算$c(S_0, K, T, r, \sigma)$和$p(S_0, K, T, r, \sigma)$。

```cpp
double blackScholesCallPrice(double K, double T, double S_t, double vol, double r)
{
    double d1 = pow((vol * sqrt(T)), -1) * (log(S_t / K) + (r + 0.5 * pow(vol, 2)) * sqrt(T));
    double d2 = d1 - vol * sqrt(T);

    //以下normcdf(x)函数3选1，默认选normcdf_poly(x)，使用normcdf_boost(x)时请自行配置boost库
    double callPrice = normcdf_poly(d1) * S_t - normcdf_poly(d2) * K * exp(-r * T); 
    //double callPrice = normcdf_integrate(d1) * S_t - normcdf_integrate(d2) * K * exp(-r * T);
    //double callPrice = normcdf_boost(d1) * S_t - normcdf_boost(d2) * K * exp(-r * T);
    return callPrice;
}
```

```cpp
double blackScholesPutPrice(double K, double T, double S_t, double vol, double r)
{
    double d1 = pow((vol * sqrt(T)), -1) * (log(S_t / K) + (r + 0.5 * vol * vol) * sqrt(T));
    double d2 = d1 - vol * sqrt(T);

    //以下normcdf(x)函数3选1，默认选normcdf_poly(x)，使用normcdf_boost(x)时请自行配置boost库
    double putPrice = normcdf_poly(-d2) * K * exp(-r * T) - normcdf_poly(-d1) * S_t;
    //double putPrice = normcdf_integrate(-d2) * K * exp(-r * T) - normcdf_integrate(-d1) * S_t;
    //double putPrice = normcdf_boost(-d2) * K * exp(-r * T) - normcdf_boost(-d1) * S_t;
    return putPrice;
}
```

#### 6.3 完整工程文件及测试

头文件*stdafx.h*

```cpp
#pragma once
#include <iostream>
#include <cmath>
//请在Visual Studio中自行配置好boost库后取消注释
//#include "boost\math\distributions\normal.hpp"
//#include <random>
```

头文件*OptionPricing.h*

```cpp
#pragma once

static const double PI = 3.14159265358979; //定义Pi值

//S_t: spotPrice
//T: maturity
//K: strikePrice
//vol: volatility
//r: riskFreeInterestRate

//声明看涨期权定价公式
double blackScholesCallPrice(double K, double T, double S_t, double vol, double r);

//声明看跌期权定价公式
double blackScholesPutPrice(double K, double T, double S_t, double vol, double r);
```

源文件*OptionPricing.cpp*

```cpp

#include "OptionPricing.h"
#include "stdafx.h"

static const double root2Pi = sqrt(2.0 * PI);

static inline double normcdf_poly(double x)
{
    if (x < 0)
        return 1 - normcdf_poly(-x);
    double k = 1 / (1 + 0.2316419 * x);
    double poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937
        + k * (-1.821255978 + 1.330274429 * k))));
    double approx = 1.0 - 1.0 / root2Pi * exp(-0.5 * x * x) * poly;
    return approx;
}

static inline double normcdf_integrate(double x)
{
    int a = 0;
    int b = 1;
    double N = 1000;
    double delta = 1 / N;
    double integrate = 0.0;
    for (int i = 0; i < N; i++)
    {
        double s = a + (i + 0.5) * delta;
        double t = x + 1 - pow(s, -1);
        double f = pow(2 * PI, -0.5) * pow(s, -2) * exp(-0.5 * t * t);
        integrate += f * delta;
    }
    return integrate;
}

//请在Visual Studio中自行配置好boost库后取消注释
/*
static inline double normcdf_boost(double x)
{
    double cdf = 0.0;

    boost::math::normal_distribution<> norm(0, 1); //生成标准正态分布
    cdf = boost::math::cdf(norm, x); //计算N(x)值
    return cdf;
}
*/

//S_t: spotPrice
//T: maturity
//K: strikePrice
//vol: volatility
//r: riskFreeInterestRate

double blackScholesCallPrice(double K, double T, double S_t, double vol, double r)
{
    double d1 = pow((vol * sqrt(T)), -1) * (log(S_t / K) + (r + 0.5 * pow(vol, 2)) * sqrt(T));
    double d2 = d1 - vol * sqrt(T);

    //以下normcdf(x)函数3选1，默认选normcdf_poly(x)，使用normcdf_boost(x)时请自行配置boost库
    double callPrice = normcdf_poly(d1) * S_t - normcdf_poly(d2) * K * exp(-r * T); 
    //double callPrice = normcdf_integrate(d1) * S_t - normcdf_integrate(d2) * K * exp(-r * T);
    //double callPrice = normcdf_boost(d1) * S_t - normcdf_boost(d2) * K * exp(-r * T);
    return callPrice;
}

double blackScholesPutPrice(double K, double T, double S_t, double vol, double r)
{
    double d1 = pow((vol * sqrt(T)), -1) * (log(S_t / K) + (r + 0.5 * vol * vol) * sqrt(T));
    double d2 = d1 - vol * sqrt(T);

    //以下normcdf(x)函数3选1，默认选normcdf_poly(x)，使用normcdf_boost(x)时请自行配置boost库
    double putPrice = normcdf_poly(-d2) * K * exp(-r * T) - normcdf_poly(-d1) * S_t;
    //double putPrice = normcdf_integrate(-d2) * K * exp(-r * T) - normcdf_integrate(-d1) * S_t;
    //double putPrice = normcdf_boost(-d2) * K * exp(-r * T) - normcdf_boost(-d1) * S_t;
    return putPrice;
}
```

测试文件*main.cpp*

```cpp
#include <iostream>
#include "OptionPricing.h"
using namespace std;

int main()
{
    double K = 100.0;
    double S_t = 110.0;
    double vol = 0.1;
    double r = 0.03;
    double T = 0.5;
    double call;
    double put;
    double putCallParity;

    call = blackScholesCallPrice(K, T, S_t, vol, r);
    cout << "The call option price is: " << call << endl; //11.6725
    put = blackScholesPutPrice(K, T, S_t, vol, r);
    cout << "The put option price is: " << put << endl; //0.183688
    cout << endl;

    cout << "<Put-Call-Parity = Call Price - Put Price> Testing" << endl;
    cout << "<Call Price - Put Price> is: " << call - put << endl;
    putCallParity = S_t - exp(-r * T) * K;
    cout << "<Put-Call-Parity> is: " << putCallParity << endl;
    cout << "Wonderful! It's correct." << endl;

    return 0;
}
```

其中看涨-看跌期权平价公式为

$$
S_0 - \exp(-rT)K = c(S_0, K, T, r, \sigma) - p(S_0, K, T, r, \sigma) \tag{6.5}
$$

### 7. 期权定价的Python实现

```python
import math
import numpy as np
from scipy.stats import norm

//定义以下简写变量
//K: strikePrice
//T: maturity
//S_t: spotPrice
//vol: volatility
//r: riskFreeRate

//定义d1和d2
def d1f(K, T, S_t, vol, r) :
    return math.pow(vol * np.sqrt(T), -1) * \
    (np.log(S_t / K) + (r + 0.5 * math.pow(vol, 2) * np.sqrt(T)))

def d2f(K, T, S_t, vol, r) :
    return d1f(K, T, S_t, vol, r) - vol * np.sqrt(T)

//定义看涨-看跌期权计算公式
def blackScholesCallPrice(K, T, S_t, vol, r) :
    d1 = d1f(K, T, S_t, vol, r)
    d2 = d2f(K, T, S_t, vol, r)

    callPrice = norm.cdf(d1) * S_t - norm.cdf(d2) * K * np.exp(-r * T)
    return callPrice

def blackScholesPutPrice(K, T, S_t, vol, r) :
    d1 = d1f(K, T, S_t, vol, r)
    d2 = d2f(K, T, S_t, vol, r)

    putPrice = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S_t
    return putPrice

//定义看涨-看跌期权平价测试公式
def callPutParity(K, T, S_t, r) :
    return S_t - np.exp(-r * T) * K

//对给定变量进行测试
K = 100.0
S_t = 110.0
vol = 0.1
r = 0.03
T = 0.5

call = blackScholesCallPrice(K, T, S_t, vol, r)
put = blackScholesPutPrice(K, T, S_t, vol, r)
callPutParity_ = callPutParity(K, T, S_t, r)

print("The call option price is: {0}".format(call))
print("The put option price is: {0}".format(put))
print("Call price - put price is: {0}".format(call - put))
print("The Call-Put-Parity is: {0}".format(callPutParity_))
```



致谢：

妈咪叔LaTeX助力 [在线LaTeX](https://www.latexlive.com)

参考:

[Stochastic Calculus for Finance II - Continuous-Time Models](https://book.douban.com/subject/2187921/)

[C++ for Financial Mathematics](https://nms.kcl.ac.uk/john.armstrong/cppbook/cpp-website.html)











