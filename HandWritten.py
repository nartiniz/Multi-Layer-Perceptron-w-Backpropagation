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

def toint(arr):
    for x in range(len(arr)):
        arr[x] = int(arr[x])
    return arr

def f(s):
    return torch.tensor(data=( 1/(1 + torch.exp(-s))),dtype=torch.float32)

def f_prime(s):
    return f(s)*(1-f(s))


ff = open("optdigits.tes", "r")

allData = [[0 for ii in range(65)] for jj in range(1797)]
allLabels = [0 for kk in range(1797)]


for _ in range(1797):
    contents = ff.readline()
    contents = contents.split(',')
    contents = toint(contents)
    # print(len(contents))
    allData[_] = contents[:64]
    allLabels[_] = contents[64]



tra_nums = [0 for p in range(10)]
test_nums = [0 for pp in range(10)]
training_data, training_labels, test_data, test_labels = [], [], [], []

for j in range(1797):
    label = allLabels[j]
    if tra_nums[label] < 40 :
        training_data.append(allData[j])
        training_labels.append(label)
        tra_nums[label] += 1
    elif test_nums[label] < 10 :
        test_data.append(allData[j])
        test_labels.append(label)
        test_nums[label] += 1


trainingLabels = [[0 for _ in range(10)]  for y in range(400)]
testLabels = [[0 for _ in range(10)]  for y in range(100)]

for x in range(len(training_labels)):
    trainingLabels[x][training_labels[x]] = 1

for x in range(len(test_labels)):
    testLabels[x][test_labels[x]] = 1




training_data = torch.tensor(training_data,dtype=torch.float32)
# training_labels = torch.tensor(training_labels,dtype=torch.float32)
test_data = torch.tensor(test_data,dtype=torch.float32)
# test_labels = torch.tensor(test_labels,dtype=torch.float32)
trainingLabels = torch.tensor(trainingLabels,dtype=torch.float32)
testLabels = torch.tensor(testLabels,dtype=torch.float32)



NN = NeuralNetwork(nb_epochs=120, mu=0.1, func=f, func_prime=f_prime)
NN.weight_initialization(layers=[64,40,35,20,10])
NN.training(training_data, trainingLabels)
prediciton = NN.prediction(test_data).numpy()
prediciton = prediciton.tolist()
print(test_labels)
label_prediction = []
for vec in prediciton:
    label_prediction.append(vec.index(max(vec)))
print(label_prediction)


plt.plot(NN.errors,color='red')
plt.ylabel("Cost")
plt.xlabel("Number of Epochs")
plt.title("Total Cost Function")
plt.show()



error = 0
for x in range(len(label_prediction)):
    if label_prediction[x] != test_labels[x]:
        error += 1
print("Classification error over 100 samples : "+ str(error))



