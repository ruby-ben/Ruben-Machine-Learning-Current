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

def xavier_initialisation(size_going_in, size_going_out, xavier_gain2):
            return xavier_gain2*np.sqrt(6.0/(size_going_in + size_going_out)) #maybe add a gain

class NeuralNetwork(nn.Module):
    def __init__(self, number_of_hidden_layers, number_of_inputs, hidden_layer_size,
                  output_layer_size, activation_function, batch_size, normalisation_coefficient, xavier_gain, device_used):
        super(NeuralNetwork, self).__init__()
        
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.normalisation_coefficient = normalisation_coefficient

        self.number_of_hidden_layers = number_of_hidden_layers   
        self.input_layer_size = number_of_inputs 
      
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.xavier_gain = xavier_gain
        self.device_used = device_used
        
     
        #torch.device('cpu')#'cuda')
        device = torch.device(device_used)#'cuda')
        #xavier init
        xavier_init_in_limit = xavier_initialisation(self.input_layer_size, self.hidden_layer_size, xavier_gain)
        xavier_init_hidden_limit = xavier_initialisation(self.hidden_layer_size, self.hidden_layer_size, xavier_gain)
        xavier_init_out_limit = xavier_initialisation(self.hidden_layer_size, self.output_layer_size, xavier_gain)

       # calcultae the std of for each network param
        with torch.no_grad():
           
            print(xavier_init_in_limit)
            print(xavier_init_hidden_limit)
            print(xavier_init_out_limit)
       
        #going to define a rank 3 weights tensor, and will make bias a rank 2 tensor 
        #first need the input to hidden layer, hidden layer-hidden layer and then hidden layer to output weights and il stack them
        self.weights_input_hidden = nn.Parameter(torch.from_numpy(np.random.uniform(low = -xavier_init_in_limit, high = xavier_init_in_limit, size = (self.input_layer_size, self.hidden_layer_size))).to(torch.float64).to(device))
        self.weights_hidden_hidden = nn.Parameter(torch.from_numpy(np.random.uniform(low = -xavier_init_hidden_limit, high = xavier_init_hidden_limit, size = (self.number_of_hidden_layers-1, self.hidden_layer_size, self.hidden_layer_size))).to(torch.float64).to(device))
        self.weights_hidden_output = nn.Parameter(torch.from_numpy(np.random.uniform(low = -xavier_init_out_limit, high = xavier_init_out_limit, size = (self.hidden_layer_size, self.output_layer_size))).to(torch.float64).to(device))
        self.bias_hidden = nn.Parameter(torch.zeros(self.number_of_hidden_layers+1, self.hidden_layer_size).to(torch.float64).to(device))
        #self.bias_hidden = nn.Parameter(torch.zeros(self.number_of_hidden_layers+1, self.hidden_layer_size).to(torch.float64).to(device))
        self.bias_output = nn.Parameter(torch.zeros(1, 1).to(torch.float64).to(device))
        
      

        
    def activation_function_normal(self, x):
        if self.activation_function == "tanh":
            activation_function_value = torch.tanh(x)

        if self.activation_function == "sigmoid":
            activation_function_value = torch.sigmoid(x)

        if self.activation_function == "GELU":
            a_value = 0.044715
            b_value = np.sqrt(2/np.pi)
            activation_function_value = 0.5*x*(1 + torch.tanh(b_value*(x + a_value*x**3)))

        return activation_function_value
    
    def forward(self, x_s): #plotting
        ai_0 = x_s.to(torch.device(self.device_used))
      
        ai_0.requires_grad_()
        wij_1 = self.weights_input_hidden
        wij_1.requires_grad_()
        bi_1 = self.bias_hidden[0].unsqueeze(1)
        zi_1= torch.matmul(wij_1.T, ai_0.T) + bi_1

        ai_1 = self.activation_function_normal(zi_1)#.view(self.hidden_layer_size, 1)
        ai_m_minus_1 = ai_1 
        zi_m = self.bias_hidden[1]  #check this
        for m in range(2, self.number_of_hidden_layers+1):
            weights_between_hidden_layers = self.weights_hidden_hidden[m-2] #checkl this it is a square matrix
            zi_m = zi_m.unsqueeze(1) + torch.matmul(weights_between_hidden_layers.T , ai_m_minus_1) #torch.Size([27, 17])
            ai_m = self.activation_function_normal(zi_m) #use a different activation fucntion like tanh symmetric
            ai_m_minus_1 = ai_m   
            zi_m = self.bias_hidden[m]
        Y_output = torch.matmul(self.weights_hidden_output.T , ai_m_minus_1) + self.bias_output

        return Y_output.T
    
class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, number_of_hidden_layers, input_layer_size, hidden_layer_size,
                  output_layer_size, device_used, dropout):
        super(RecurrentNeuralNetwork, self).__init__()
        
        self.number_of_hidden_layers = number_of_hidden_layers   
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.device_used = device_used

        self.LSTM = nn.LSTM(input_layer_size, hidden_layer_size, number_of_hidden_layers, batch_first = True, dropout=dropout)# may have to chance this back to change the format of the input
        self.fc = nn.Linear(hidden_layer_size, output_layer_size)
        #add initialisation of weights for better convergence
        self._initialize_weights()

    def forward(self, x): #plotting
        #initialize the hidden state with zeros
        h0 = torch.zeros(self.number_of_hidden_layers,x.size(0), self.hidden_layer_size).to(self.device_used)
        c0 = torch.zeros(self.number_of_hidden_layers,x.size(0), self.hidden_layer_size).to(self.device_used)
        
        #rnn forward pass
        out, _ = self.LSTM(x, (h0,c0)) #input format is (batch_size, sequence_length, input_size)

        #decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out
    def _initialize_weights(self):
        for name, param in self.LSTM.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))