---
title: "我的第一篇深度学习笔记"
date: 2026-06-07
draft: false
math: true
tags: ["深度学习", "入门"]
---

这是我的第一篇文章。来测试一下公式和代码。

## 数学公式测试

行内公式：损失函数 $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$。

独立公式：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

## 代码测试

```python
import torch
x = torch.randn(3, 3)
print(x.softmax(dim=-1))
```
