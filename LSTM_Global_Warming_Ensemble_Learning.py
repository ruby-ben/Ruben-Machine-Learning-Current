#importing libaries and initialising my network
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import plotly.graph_objects as go
#from plotly.colors import n_colorsy
import Deep_Neural_Network_Script
import mpmath
from sklearn.preprocessing import MinMaxScaler

torch.cuda.empty_cache()

#prepare data
file_name_I1 = 'GlobalTemperatures.csv'
I1_data = pd.read_csv(file_name_I1)
Cummulative_days = I1_data['days ']
Mean_temp = I1_data['Land Average Temperature']
Mean_temp_uncertainty = I1_data['Temp uncertainty']

#Averaging over 12 months 
Cleaned_mean_temp = []
Cleaned_cummulative_days = []
for i in range(int(len(Mean_temp)/12)):
    Cleaned_mean_temp.append(sum(Mean_temp[12*i: 12*(i+1)])/12)
    Cleaned_cummulative_days.append(sum(Cummulative_days[12*i: 12*(i+1)])/12)
#disregarding pre 1850 data due to unreliablily of the equipment of the time
Cleaned_cummulative_days = np.array(Cleaned_cummulative_days[128:]) #128
Cleaned_mean_temp = np.array(Cleaned_mean_temp[128:])

#normalise my data
normalised_Cummulative_days = (Cleaned_cummulative_days-min(Cleaned_cummulative_days))/(max(Cleaned_cummulative_days)-min(Cummulative_days))
normalised_Mean_temp = (Cleaned_mean_temp-min(Cleaned_mean_temp))/(max(Cleaned_mean_temp)-min(Cleaned_mean_temp))

#trained models 
saved_model1 = 'LSTM_2x500_run1_part3.pth'
saved_model2 = 'LSTM_2x500_run3.pth'
saved_model3 = 'LSTM_2x500_run2.pth'

#get the LSTM class
poly_model1 = Deep_Neural_Network_Script.RecurrentNeuralNetwork(
    number_of_hidden_layers = 2, #5, 100
    input_layer_size = 1,  
    hidden_layer_size = 500, #500
    output_layer_size = 1, 
    device_used = 'cpu',
    dropout=0.1
)

poly_model2 = Deep_Neural_Network_Script.RecurrentNeuralNetwork(
    number_of_hidden_layers = 2, #5, 100
    input_layer_size = 1,  
    hidden_layer_size = 500, #500
    output_layer_size = 1, 
    device_used = 'cpu',
    dropout=0.1
)

poly_model3 = Deep_Neural_Network_Script.RecurrentNeuralNetwork(
    number_of_hidden_layers = 2, #5, 100
    input_layer_size = 1,  
    hidden_layer_size = 500, #500
    output_layer_size = 1, 
    device_used = 'cpu',
    dropout=0.1
)

#load trained models
poly_model1.load_state_dict(torch.load(saved_model1))
poly_model2.load_state_dict(torch.load(saved_model2))
poly_model3.load_state_dict(torch.load(saved_model3))

#Ensemble average
Ensemble_average = (poly_model1.forward(torch.tensor(normalised_Cummulative_days, dtype = torch.float32).reshape(-1,1,1)) + poly_model2.forward(torch.tensor(normalised_Cummulative_days, dtype = torch.float32).reshape(-1,1,1))  + poly_model3.forward(torch.tensor(normalised_Cummulative_days, dtype = torch.float32).reshape(-1,1,1)))/3

#Plot results

#Important historical dates for innovation
#1879 - Invention of the lightbulb
#1908 - Mass production of automobiles
#1930 - Developement of synthetic chemicals (CFCs)
#1945 - Rise of the oil industry
#1950 - Expansion of the Suburbs
#1960 - Introduction of jet engines
#1970 - Introduction of coal fired power plants
#1980 - Advent of computer and digital revolution
#2000 - Expansion of the internet and Data centres (internet, cloud computing etc)
#2010 - Rise of cryptocurrencies
historical_innovation = [1879,1908,1930,1945,1950,1960,1970,1980,2000,2010]
innovation_labels = ['Industrial Revolution', 'Railways/steamships', 'Lightbulb', 'Automobiles', 
                     'CFCs', R'Oil industry', 'Suburbanization', 'Jet engines', 'Coal fired power plants', 'Computer revolution', 'Internet expansion', 'Cryptocurrencies']


#renormalise the model #convert the cumulatibe_days to years
plt.plot(Cleaned_cummulative_days/365 + 1750 , Ensemble_average.detach().numpy()*(max(Cleaned_mean_temp) -  min(Cleaned_mean_temp)) + min(Cleaned_mean_temp), color = "green", label = "LSTM", linewidth = 2)
plt.plot(Cleaned_cummulative_days/365 + 1750 , Cleaned_mean_temp, color = 'blue', label = "actual temp", linewidth = 2)

#plt.plot(Cleaned_cummulative_days/365 + 1750 , poly_model.forward(torch.tensor(normalised_Cummulative_days, dtype = torch.float32).reshape(-1,1,1)).detach().numpy(), color = "green", label = "LSTM")
#plt.plot(Cleaned_cummulative_days/365 + 1750 , normalised_Mean_temp, color = 'blue', label = "actual temp")

colors = ['red']
for x, label in zip(historical_innovation, innovation_labels):#, colors):
    plt.axvline(x=x, color='red', linestyle='--', linewidth=1, label = label)
plt.legend(loc = (1.01,0.0))
plt.title('LSTM (2x500, dropout=0.1, sequence_length=3)')
plt.xlabel("Years")
plt.ylabel("Mean Temperature of Earth's Surface(\u00B0C)") #unicode symbol

plt.savefig('plot.png', bbox_inches='tight')
