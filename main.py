import torch
import numpy as np
import model

m = model.CNN()
x = np.ones((1, 1, 3, 3))
x = torch.Tensor(x)
y = m.forward(x)
print(y)