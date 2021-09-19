+++
title = "Chapter 03 Brownian Motion"
date = "2021-09-19"
+++


## 3.2 Scaled Random Walks


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
```


```python
#np.random.seed(42)
```

### 3.2.1 Symmetric Random Walk


```python
t = 10
steps = np.arange(t)
random_walks = np.random.randint(0, 2, size=t) * 2 - 1
random_walks.shape
```




    (10,)




```python
print(random_walks.mean())
print(random_walks.std())
```

    -0.4
    0.9165151389911679



```python
random_walks = random_walks.cumsum()
```


```python
def plot_random_walks(step, random_walk):
    plt.plot(step, random_walk)
    plt.xlabel("Time/Steps", fontsize=14)
    plt.ylabel("Location", fontsize=14)
    plt.grid()
```


```python
plt.figure(figsize=(8, 6))
plot_random_walks(steps, random_walks)
```


​    
![png](output_9_0.png)
​    


### 3.2.5 Scaled Symmetric Random Walk


```python
n = 100
```


```python
steps = np.arange(0, t, 1/n)
```


```python
random_walks = 1 / np.sqrt(n) * (np.random.randint(0, 2, size=n*t) * 2 - 1)
random_walks.shape
```




    (1000,)




```python
print(random_walks.mean())
print(random_walks.var())
```

    0.0017999999999999989
    0.00999676



```python
random_walks = random_walks.cumsum()
random_walks.shape
```




    (1000,)




```python
plt.figure(figsize=(8, 6))
plot_random_walks(steps, random_walks)
```


​    
![png](output_16_0.png)
​    



```python
nwalks = 20
random_walks = 1 / np.sqrt(n) * (np.random.randint(0, 2, size=(nwalks, len(steps))) * 2 - 1)
```


```python
random_walks = random_walks.cumsum(axis=1)
random_walks.shape
```




    (20, 1000)




```python
plt.figure(figsize=(8, 6))
for i in range(len(random_walks)):
    plot_random_walks(steps, random_walks[i])
plt.grid()
```


​    
![png](output_19_0.png)
​    


### 3.2.6 Limiting Distribution of the Scaled Random Walk


```python
n = 1000
steps = np.arange(0, t, 1/n)
nwalks = 10000

random_walks = 1 / np.sqrt(n) * (np.random.randint(0, 2, size=(nwalks, len(steps))) * 2 - 1)
random_walks = random_walks.cumsum(axis=1)
random_walks.shape
```




    (10000, 10000)




```python
random_walks_final = random_walks[:, len(steps)-1]
```


```python
plt.figure(figsize=(8, 6))
plt.hist(random_walks_final, bins=20)
plt.xlabel("Final location", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid()
```


​    
![png](output_23_0.png)
​    



```python
print(random_walks_final.mean())
print(random_walks_final.var())
```

    0.03362133608291024
    9.684154405760037


### 3.2.7 Log-Normal Distribution as the Limit of the Binomial Model

假设

\\[
r = 0
\\]

\\[
u\_n = 1 + \\frac{\\sigma}{\\sqrt{n}}
\\]

\\[
d\_n = 1 - \\frac{\\sigma}{\\sqrt{n}}
\\]

\\[
\\tilde{p} = \\frac{1 + r - d\_n}{u\_n - d\_n} = \\frac{\\frac{\\sigma}{\\sqrt{n}}}{2 \\frac{\\sigma}{\\sqrt{n}}} = \\frac{1}{2}
\\]

\\[
\\tilde{q} = \\frac{u\_n - 1 - r}{u\_n - d\_n} = \\frac{\\frac{\\sigma}{\\sqrt{n}}}{2 \\frac{\\sigma}{\\sqrt{n}}} = \\frac{1}{2}
\\]

则

\\[
nt = H\_{nt} + T\_{nt}
\\]

\\[
M\_{nt} = H\_{nt} - T\_{nt}
\\]

进而可以求的

\\[
H\_{nt} = \\frac{1}{2} (nt + M\_{nt})
\\]

\\[
T\_{nt} = \\frac{1}{2} (nt - M\_{nt})
\\]

股价\\(S\_n(t)\\)可以表示为\\(S(0)\\)为起点，上升\\(H\_{nt}\\)次和下降\\(T\_{nt}\\)次以后的金额即：

\\[
S\_n(t) = S(0) u\_n ^ {H\_{nt}} d\_n ^ {T\_{nt}} = S(0) \\ \\left( 1 + \\frac{\\sigma}{\\sqrt{n}} \\right) ^ {\\frac{1}{2} (nt + M\_{nt})} \\left( 1 - \\frac{\\sigma}{\\sqrt{n}} \\right) ^ {\\frac{1}{2} (nt - M\_{nt})}
\\]


```python
n = 100
t = 10

r = 0
sigma = 0.6
S_0 = 100
u_n = 1 + sigma/np.sqrt(n*t)
d_n = 1 - sigma/np.sqrt(n*t)
p = 0.5
q = 0.5
```


```python
steps = np.arange(0, t, 1/n)
M = np.random.randint(0, 2, size=n*t) * 2 - 1
M = M.cumsum()
M.shape
```




    (1000,)




```python
h_nt = 0.5 * (n*t + M)
t_nt = 0.5 * (n*t - M)
```


```python
S_t = S_0 * pow(u_n, h_nt) * pow(d_n, t_nt)
```


```python
plt.figure(figsize=(8, 6))
plot_random_walks(steps, S_t)
```


​    
![png](output_32_0.png)
​    



```python
from numpy import random

#生成risk-neutral probability
rnp = random.choice([u_n, d_n], p=[p, q], size=(nwalks, n*t))
rnp_cumprod = rnp.cumprod(1)

paths = rnp_cumprod[:, n*t-1]
paths = paths * S_0
```


```python
bins = 40
plt.figure(figsize=(8, 6))
plt.hist(paths, bins = bins)
plt.xlabel("Final location", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid()
```


​    
![png](output_34_0.png)
​    



```python
sample_mean = paths.mean()
theoretical_mean = S_0 * np.exp(r)

sample_std = paths.std()
theoretical_std = np.sqrt(S_0 * S_0 * np.exp(2*r) * (np.exp(sigma * sigma) - 1))
```


```python
print("sample_mean = %f" % sample_mean)
print("theoretical_mean = %f" % theoretical_mean)
print("sample_std = %f" % sample_std)
print("theoretical_std = %f" % theoretical_std)
```

    sample_mean = 99.527160
    theoretical_mean = 100.000000
    sample_std = 64.237271
    theoretical_std = 65.827761



```python
plt.figure(figsize=(8, 6))
plt.hist(np.log(paths), bins = bins)
plt.xlabel("Final location", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid()
```


​    
![png](output_37_0.png)
​    



```python
log_paths = np.log(paths)
```


```python
log_paths.mean()
```




    4.423820847745468




```python
log_paths.var()
```




    0.355568197550482




```python
sigma * sigma * t
```




    3.5999999999999996




```python

```
