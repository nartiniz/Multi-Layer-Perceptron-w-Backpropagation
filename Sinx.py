import numpy as np
import cv2,torch,pickle,math
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
import torch.utils.data
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import math
from EE550HW4 import NeuralNetwork


def f(s):
    return torch.tensor(data=1-2/(1 + torch.exp(s)),dtype=torch.float32)

def f_prime(s):
    return torch.tensor(data=2/(2 + torch.exp(s)+torch.exp(-s)),dtype=torch.float32)


NN = NeuralNetwork(nb_epochs=500, mu=0.1, func=f, func_prime=f_prime)
NN.weight_initialization(layers=[1,10,7,6,1])
x = torch.empty(size=(100,1)).uniform_(0,2*math.pi)
y = torch.sin(x)
NN.training(x,y)

test = torch.empty(size=(25,1)).uniform_(0,2*math.pi)
test_out = NN.prediction(test)
real_out = torch.sin(test)
print(real_out)
print(test_out)



plt.plot(NN.errors,color='red')
plt.ylabel("Cost")
plt.xlabel("Number of Epochs")
plt.title("Total Cost Function")
plt.show()

kk = np.arange(start=0,stop=2*math.pi,step=0.01)
y = []
for x in kk:
    y.append(math.sin(x))
y = np.array(y)
plt.plot(kk,y,color='red')
plt.scatter(test.numpy(),test_out.numpy(),marker='*',linewidths=0.5)
plt.legend(["Sin(x)", "Test Results"])
plt.title("Sin(x) Approximations")
plt.ylabel("sin(x)")
plt.xlabel("x")
plt.xticks(np.arange(0, 2*math.pi+math.pi/2, step=math.pi/2), ('0', '\u03C0/2', '\u03C0', '3\u03C0/2', '2\u03C0'))
plt.grid()
plt.show()








