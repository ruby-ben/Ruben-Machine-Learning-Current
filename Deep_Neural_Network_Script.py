#train a large s-x randomly generated space.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd
#import h5py

#functions 

def sigmoid_3rd_derivative(x):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)*(1-6*sigmoid_x + 6*sigmoid_x**2)


def sigmoid_2nd_derivative(x):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)*(1-2*sigmoid_x)

def sigmoid_derivative(x):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def test_function(x1, x2, x3, mH2, s12, s14):
    mt2 = 1
    F1 = mt2 + 2*mt2*(x1 + x2 + x3) + mt2*(x1**2 + x2**2 + x3**2) + 2*mt2*(x2*x3 + x1*x3 + x1*x2) - mH2*(x1*x2 + x2*x3) - x2*s14 - x1*x3*s12
    F1 = 1/F1# return x1**2 + x2*x1
    return F1**2

def new_test_function(t1, t2, t3, mH2, s12, s14):
    mt2 = 1
    x1 = (3-2*t1)*t1**2
    x2 = (3-2*t2)*t2**2
    x3 = (3-2*t3)*t3**2

    F1 = mt2 + 2*mt2*(x1 + x2 + x3) + mt2*(x1**2 + x2**2 + x3**2) + 2*mt2*(x2*x3 + x1*x3 + x1*x2) - mH2*(x1*x2 + x2*x3) - x2*s14 - x1*x3*s12
    F1 = 1/F1# return x1**2 + x2*x1

    w1 = 6*t1*(1-t1)
    w2 = 6*t2*(1-t2)
    w3 = 6*t3*(1-t3)

    return w1*w2*w3*F1**2

def normalised_test_function(x1, x2, x3, mH2, s12, s14):
    normalised_value = new_test_function(x1, x2, x3, mH2, s12, s14)/new_test_function(0.5, 0.5, 0.5, mH2, s12, s14)
    return x1*x2*x3*mH2*s12*s14#normalised_value

#def dummy_function(t1, t2, t3, mH2, s12, s14):
 #   x1 = (3-2*t1)*t1**2
  #  x2 = (3-2*t2)*t2**2
   # x3 = (3-2*t3)*t3**2

    
 #   w1 = 6*t1*(1-t1)
  #  w2 = 6*t2*(1-t2)
   # w3 = 6*t3*(1-t3)
    

   # return w1*w2*w3*x1*x2*x3*mH2*s12*s14#normalised_value


def dummy_function(t1, t2, t3, mH2, s12, s14):
    x1 = (3-2*t1)*t1**2
    x2 = (3-2*t2)*t2**2
    x3 = (3-2*t3)*t3**2

    
    w1 = 6*t1*(1-t1)
    w2 = 6*t2*(1-t2)
    w3 = 6*t3*(1-t3)

    numerator = s12*s14*x1*x2 + mH2*x3 
    denominator = 0.25*s12*s14 + 0.5*mH2

    dummy_function_tilda = numerator/denominator


    return w1*w2*w3*(dummy_function_tilda)#normalised_value

def xavier_initialisation(size_going_in, size_going_out):
    return np.sqrt(6.0/(size_going_in + size_going_out))

#Network architecture
#number_of_auxillary_variable = 3
#number_of_phase_space_parameters = 3
#input_layer_size = number_of_auxillary_variable+number_of_phase_space_parameters
#hidden_layer_size = 50
#output_layer_size = 1




class NeuralNetwork(nn.Module):
    def __init__(self, number_of_hidden_layers, number_of_auxillary_variable, number_of_phase_space_parameters, hidden_layer_size, output_layer_size):
        super(NeuralNetwork, self).__init__()
        self.number_of_hidden_layers = number_of_hidden_layers   
        self.number_of_auxillary_variable = number_of_auxillary_variable  
        self.number_of_phase_space_parameters = number_of_phase_space_parameters
        self.input_layer_size = self.number_of_auxillary_variable + self.number_of_phase_space_parameters
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        #torch.device('cpu')#'cuda')


        #xavier init
        xavier_init_in_limit = xavier_initialisation(self.input_layer_size, self.hidden_layer_size)
        xavier_init_hidden_limit = xavier_initialisation(self.hidden_layer_size, self.hidden_layer_size)
        xavier_init_out_limit = xavier_initialisation(self.hidden_layer_size, self.output_layer_size)
        print(xavier_init_in_limit)
        print(xavier_init_hidden_limit)
        print(xavier_init_out_limit)


        #going to define a rank 3 weights tensor, and will make bias a rank 2 tensor 
        #first need the input to hidden layer, hidden layer-hidden layer and then hidden layer to output weights and il stack them
        self.weights_input_hidden = nn.Parameter(torch.from_numpy(np.random.uniform(low = -xavier_init_in_limit, high = xavier_init_in_limit, size = (self.input_layer_size, self.hidden_layer_size))).to(torch.float32))
        self.weights_hidden_hidden = nn.Parameter(torch.from_numpy(np.random.uniform(low = -xavier_init_hidden_limit, high = xavier_init_hidden_limit, size = (self.number_of_hidden_layers-1, self.hidden_layer_size, self.hidden_layer_size))).to(torch.float32))
        self.weights_hidden_output = nn.Parameter(torch.from_numpy(np.random.uniform(low = -xavier_init_out_limit, high = xavier_init_out_limit, size = (self.hidden_layer_size, self.output_layer_size))).to(torch.float32))

        self.bias_hidden = nn.Parameter(torch.zeros(self.number_of_hidden_layers+1, self.hidden_layer_size))
        #bias from hidden to output layer doesn't get trained
        #print(self.weights_hidden_hidden[0].shape) is hidden1 to hidden2

        

    def forward(self, x, s): #plotting

        ai_0 = torch.cat((x, s), dim = 0)
        wij_1 = self.weights_input_hidden
        bi_1 = self.bias_hidden[0]  
        zi_1 = bi_1
        for i in range(0,  self.input_layer_size): 
            zi_1 = zi_1 + ai_0[i]*wij_1[i]

        ai_1 = torch.sigmoid(zi_1).T

        #set the starting point of the math
        ai_m_minus_1 = ai_1 #we will constanstly update this value when movig to the next layers m
        zi_m = self.bias_hidden[1] #got to get the bias for the first hidden layer to the second hidden layer

        
        for m in range(2, self.number_of_hidden_layers+1):
                 #nth compeontx
            for n in range(0, self.hidden_layer_size): 
                 #summation a function
                zi_m = zi_m + ai_m_minus_1[n]*self.weights_hidden_hidden[m-2][n]

            ai_m = torch.sigmoid(zi_m)  
                #adjust the notation for the next loop 
            ai_m_minus_1 = ai_m    
            zi_m = self.bias_hidden[m]# is this m-1

        Y_output = torch.matmul(ai_m_minus_1, self.weights_hidden_output)
       # print(Y_output.shape)

        return Y_output
    
  
