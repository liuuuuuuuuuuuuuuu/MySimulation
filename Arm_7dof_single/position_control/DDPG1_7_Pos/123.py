import numpy as np
import copy
import torch

a = torch.tensor([1,2,3,4])
# b = a.copy()
b= a
b.data[1]=100
print(a,b)