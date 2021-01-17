# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:05:57 2021

@author: lankuohsing
"""
import torch
# In[]

x1=torch.tensor(1, requires_grad=True, dtype = torch.float)
x2=torch.tensor(2, requires_grad=True, dtype = torch.float)
x3=torch.tensor(3, requires_grad=True, dtype = torch.float)
y=torch.randn(3)
y[0]=x1**3+2*x2**2+3*x3
y[1]=3*x1+2*x2**2+x3**3
y[2]=2*x1+x2**3+3*x3**2
v=torch.tensor([3,2,1],dtype=torch.float)
y.backward(v)
print(x1.grad)
print(x2.grad)
print(x3.grad)
