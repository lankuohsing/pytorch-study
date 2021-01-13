# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:43:53 2020

@author: lankuohsing
"""
from __future__ import print_function
import torch
# In[]
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
# In[]
Q = 3*a**3 - b**2
# In[]
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
#Q.sum().backward()#can replace the code above
# In[]
# check if collected gradients are correct
# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)