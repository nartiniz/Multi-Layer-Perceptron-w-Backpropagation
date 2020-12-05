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
    return torch.tensor(data=( 1/(1 + torch.exp(-s))),dtype=torch.float32)

def f_prime(s):
    return f(s)*(1-f(s))


NN = NeuralNetwork(nb_epochs=10000, mu=0.1, func=f, func_prime=f_prime)
NN.weight_initialization(layers=[2,10,1])
x = torch.tensor(data=np.array([[1,1],[1,0],[0,1],[0,0]]), dtype=torch.float32)
y = torch.tensor(data=np.array([0.,1.,1.,0.]))
NN.training(x,y)    
test = torch.tensor(data=([[1,1],[1,0],[0,1],[0,0]]),dtype=torch.float32)
print(NN.prediction(test))
# print(NN.errors)


plt.plot(NN.errors,color='red')
plt.ylabel("Cost")
plt.xlabel("Number of Epochs")
plt.title("Total Cost Function")
plt.show()




