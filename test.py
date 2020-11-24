# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,4"
print(torch.cuda.device_count())
x = torch.rand(5, 3)
print(x)

import matplotlib.pyplot as plt

a = range(20)
b = range(20)
plt.plot(a,b)
plt.show()

# The output should be something similar to:
# tensor([[0.2989, 0.2493, 0.2642],
#         [0.9508, 0.4811, 0.1085],
#         [0.5423, 0.3216, 0.3068],
#         [0.8863, 0.8385, 0.5150],
#         [0.8451, 0.4620, 0.6266]])