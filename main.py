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


class NeuralNetwork:
    def __init__(self, nb_epochs, mu, func, func_prime):
        self.weights, self.delta_weights,self.lambdas,self.s_values,self.o_values = [], [], [], [], []
        self.nb_epochs, self.mu, self.function, self.function_prime = nb_epochs, mu, func, func_prime
        self.errors = []
    def weight_initialization(self,layers):
        for j  in range(len(layers)-1):
            self.weights.append(torch.empty(size=(layers[j], layers[j+1])).uniform_(-1,1))
            self.delta_weights.append(torch.zeros(size=(layers[j], layers[j+1]), dtype=torch.float32))
        for i in range(len(layers)):
            self.lambdas.append(torch.zeros(size=(layers[i],1), dtype=torch.float32))
            self.s_values.append(torch.zeros(size=(layers[i],1), dtype=torch.float32))
            self.o_values.append(torch.zeros(size=(layers[i],1), dtype=torch.float32))

    def forward(self,x):
        self.o_values[0] = x
        self.s_values[0] = x
        for ind in range(len(self.weights)):
            x = torch.matmul(x,self.weights[ind])
            self.s_values[ind+1] = x
            x = self.function(x)
            self.o_values[ind+1] = x
        return x

    def Error(self,y):
        out = self.o_values[-1]
        err = (1/2)*((out-y).pow(2).sum())
        err = err.numpy()
        return err

    def lambda_backpropagation(self,y):
        self.lambdas[-1] = -1*(-(y-self.o_values[-1]))*self.function_prime(self.s_values[-1])
        for index in range(len(self.weights)-1,-1,-1):
            self.lambdas[index] = torch.matmul(self.weights[index],self.lambdas[index+1]) * self.function_prime(self.s_values[index])


    def backpropagation(self,y,mu):
        self.lambda_backpropagation(y)
        for ind in range(len(self.weights)):
            o_shape = self.o_values[ind].numpy().shape[0]
            l_shape = self.lambdas[ind+1].numpy().shape[0]
            self.delta_weights[ind] = mu * torch.matmul(self.o_values[ind].reshape(shape=(o_shape,1)), self.lambdas[ind+1].reshape(shape=(1,l_shape)))

    def update_weights(self):
        for k in range(len(self.weights)):
            # self.weights[k] += self.delta_weights[k]
            self.weights[k] = torch.add(self.weights[k],self.delta_weights[k])
    def training(self,training_data,training_out):
        for _ in range(self.nb_epochs):
            err = 0
            for z in range(len(training_data)):
                x, y = training_data[z], training_out[z]
                self.forward(x)
                err += self.Error(y)
                # print(str(_)+" : " +str(self.Error(y)))
                # self.lambda_backpropagation(y)
                self.backpropagation(y,self.mu)
                self.update_weights()
            self.errors.append(err)

    def prediction(self,test):
        pred = []
        for u in range(len(test)):
            pred.append(self.forward(test[u]).numpy())
        return torch.tensor(pred,dtype=torch.float32)


