# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 2022
@author: Abhilash Adya

Note: 1) Please assign the path where you have stored the dataset(.csv file) in 'path' variable below.
      2) One can see the graph of any column like 'new_cases' and 'population' vs year_week just by writing down 
         the name of the column in variable 'y_cordinate'. Here, we have used "tests_done".
"""




# importing necessary libraries/functions/classes.

import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Assigning Variables

y_cordinate = "new_cases"
x_cordinate = 'year_week'
path = os.path.join(os.getcwd(),"Original_data.csv")
win_size = 30
batch_size = 10
warnings.filterwarnings('ignore')

# Reading given data

data = pd.read_csv(path)


# Checking for null datasets

data[y_cordinate].isnull().values.any()
data[x_cordinate].isnull().values.any()

# Removing the null values from data

data = data.dropna(axis=0, subset=[x_cordinate, y_cordinate])
#print(data[y_cordinate].isnull().values.any())


# Using groupby function to sum the data for all the given countries with respect to year_week column.
var = dict(data.groupby(x_cordinate)[y_cordinate].sum())

# Converting obtained dictionary into dataframe
data_for_scaling = pd.DataFrame(var.items(), columns=[x_cordinate, y_cordinate])


# Converting string values of time(here x_cordinate) into timestamps

time_list = data_for_scaling['year_week'].tolist()
time_stamps = []
for element in time_list:
 r = datetime.datetime.strptime(element + '-1', "%Y-W%W-%w")
 time_stamps.append(r)



data_for_scaling = data_for_scaling.drop(columns=['year_week'])
data_for_scaling['time_stamps'] =  time_stamps

data_for_scaling['datetime'] = pd.to_datetime(data_for_scaling['time_stamps'])
data_for_scaling.set_index('datetime', inplace=True)
data_for_scaling = data_for_scaling.drop(columns=['time_stamps'])

all_data = data_for_scaling[data_for_scaling.columns[0]].values.tolist()


# Standardization: 0 mean and 1 standard deviation

df_scaled = (data_for_scaling-data_for_scaling.mean())/data_for_scaling.std()
    

#print(df_new)

scaled_values = y_cordinate
scaled_values = df_scaled[df_scaled.columns[0]].values.tolist()

#print(scaled_values)

# Putting the obtained values in a dataframe, because it is easy to plot a dataframe

df = pd.DataFrame(list(zip(time_stamps, scaled_values)),
              columns=['datetime',y_cordinate])

df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

datetime_list = df.index.tolist()


def windowing(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

windowed_dataset = windowing(scaled_values,win_size)
X = []
Y = []
for window in windowed_dataset:
    X.append(window[0])
    Y.append(window[1])
X = np.array(X)
y = np.array(Y)

windowed_datetime_dataset = windowing(datetime_list,win_size)
X_time = []
Y_time = []
for elements in windowed_datetime_dataset:
    X_time.append(elements[0])
    Y_time.append(elements[1])
X_datetime = np.array(X_time)
y_datetime = np.array(Y_time)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle= False, random_state=42)
X_datetime_train, X_datetime_test, y_datetime_train, y_datetime_test = train_test_split(X_datetime, y_datetime, test_size=0.2, shuffle = False)
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,shuffle= False, random_state=42)
X_datetime_train, X_datetime_val, y_datetime_train, y_datetime_val = train_test_split(X_datetime_train, y_datetime_train, test_size=0.25, shuffle = False)
#print(X_datetime_test)

# Convert the data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_val_tensor = torch.tensor(X_val, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.float)
y_val_tensor = torch.tensor(y_val, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)

# Create a dataset for each set
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create a data iterator for each dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to("cpu")

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out
    
# Getting the aspired model
    
def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
    }
    return models.get(model.lower())(**model_params)

# Model Training

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()
    
    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        #model_path = f'{self.model}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        #script_module = torch.jit.script(model)
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to("cpu")
                y_batch = y_batch.to("cpu")
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to("cpu")
                    y_val = y_val.to("cpu")
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            #if (epoch <= 10) | (epoch % 50 == 0):
            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
            )

        #torch.save(self.model.state_dict(), model_path)
        #torch.jit.save(script_module, "Optimization.pt")
    def evaluate(self, test_loader, batch_size=10, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to("cpu")
                y_test = y_test.to("cpu")
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to("cpu").detach().numpy())
                values.append(y_test.to("cpu").detach().numpy())

        return predictions, values
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("RNN Losses")
        plt.show()
        plt.close()
   
input_dim = 2
output_dim = 1
hidden_dim = 12
layer_dim = 3
batch_size = batch_size
dropout = 0.2
n_epochs = 15
learning_rate = 1e-3
weight_decay = 1e-6


model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model = get_model('rnn', model_params)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=2)
opt.plot_losses()

predictions, values = opt.evaluate(test_loader, batch_size=batch_size, n_features=input_dim)



def reverse_scaling(pred,standard_deviation,mean):
    unscaled=[]
    for val in pred:
        val = (val * standard_deviation) + mean
        unscaled.append(val)
    return unscaled

std = data_for_scaling[y_cordinate].std()   
mean = data_for_scaling[y_cordinate].mean()
  
prediction_unscaled = reverse_scaling(predictions,std,mean)
values_unscaled = reverse_scaling(values,std,mean)

# Converting numpy arrays to float using Itertools library
prediction_unscaled_float = list(itertools.chain.from_iterable([val.flatten() for val in prediction_unscaled]))
values_unscaled_float = list(itertools.chain.from_iterable([val.flatten() for val in values_unscaled]))

datetime = y_datetime_test.tolist()
datetime = [item for sublist in y_datetime_test for item in sublist]



df_result = pd.DataFrame(list(zip(datetime, prediction_unscaled_float, values_unscaled_float)),
              columns=['datetime','prediction', y_cordinate])



# Error calculation

# =============================================================================
# def mean_squared_error(y_true, y_pred):
#     return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
# 
# mse = mean_squared_error(values_unscaled,prediction_unscaled)
# print("Mean Squarred Error:",mse)
# 
# def mean_absolute_error(y_true, y_pred):
#     return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
# 
# mae = mean_absolute_error(values_unscaled,prediction_unscaled)
# print("Mean Absolute Error:", mae)
# 
# =============================================================================
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
rmse = root_mean_squared_error(values_unscaled,prediction_unscaled)
print("Root Mean Squared Error For RNN:", rmse)


####################            THETA MODEL           ########################################################

# Seasonality test
plot_acf(data_for_scaling[y_cordinate], lags=100)
plt.xlabel("Week",fontweight='bold', fontsize=14)


additive_decomposition = seasonal_decompose(data_for_scaling[y_cordinate], model='additive', period=7)

plt.rcParams.update({'figure.figsize': (12,6)})
additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.1, 0.95, 0.95])
plt.xlabel("Week",fontweight='bold', fontsize=14)


seasonality=additive_decomposition.seasonal
seasonality.plot(color='green') 
plt.title("Seasonality", fontsize=16)
plt.xlabel("Week",fontweight='bold', fontsize=14)

# Stationarity Test
result = adfuller(data_for_scaling[y_cordinate])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))
 
    
 
# Creating windows    
windowed_dataset = windowing(all_data,win_size)

X_list = []
Y_list = []
for window in windowed_dataset:
    X_list.append(window[0])
    Y_list.append(window[1])

# Converting the data for prediction into numpy arrays
X = np.array(X_list)
y = np.array(Y_list)

windowed_datetime_dataset = windowing(datetime_list,win_size)
X_time = []
Y_time = []
for elements in windowed_datetime_dataset:
    X_time.append(elements[0])
    Y_time.append(elements[1])
X_datetime = np.array(X_time)
y_datetime = np.array(Y_time)    

# Splitting the dataset into training(60%), testin(20%) and validation(20%):
    
# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle= False, random_state=42)
X_datetime_train, X_datetime_test, y_datetime_train, y_datetime_test = train_test_split(X_datetime, y_datetime,shuffle= False, test_size=0.2, random_state=42)

# Creating a validation set by splitting training set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,shuffle= False, random_state=42)
X_datetime_train, X_datetime_val, y_datetime_train, y_datetime_val = train_test_split(X_datetime_train, y_datetime_train, test_size=0.25, random_state=42)


# Creating tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_val_tensor = torch.tensor(X_val, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.float)
y_val_tensor = torch.tensor(y_val, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)

# Create a dataset for each training,testing and validation sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Loading into the dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)


### Modeling

# Function for Theta model   
def thetamodel_train(train_len, horizon, loader):
    pred_theta_test = []
    loss=[]
    horizon = int(horizon)
    for i, (X_batch, y_batch) in enumerate(loader):
       for j in range(X_batch.shape[0]):
            input_batches= X_batch[j]    
            tm = ThetaModel(endog = input_batches, period=7)
            res = tm.fit()
            predictions = res.forecast(horizon)
            pred_theta_test.extend(predictions)
            batch_loss = root_mean_squared_error(y_batch, predictions[:len(y_batch)])
            loss.append(batch_loss)
            print(" Training loss for theta:", batch_loss)
    mean_loss = np.mean(loss)
    print("Total training Loss",mean_loss)
    return loss,pred_theta_test

# To get the predictions:

# Training set predictions 
training = len(X_train_tensor)
train_horizon = len(y_train_tensor)

pred_theta_train = thetamodel_train(training, train_horizon, train_loader)


# Validation set predictions
validation = len(X_val_tensor)
val_horizon = len(y_val_tensor)

def thetamodel_val(train_len, horizon, loader):
    pred_theta_test = []
    loss=[]
    horizon = int(horizon)
    for i, (X_batch, y_batch) in enumerate(loader):
       for j in range(X_batch.shape[0]):
            input_batches= X_batch[j]    
            tm = ThetaModel(endog = input_batches, period=7)
            res = tm.fit()
            predictions = res.forecast(horizon)
            pred_theta_test.extend(predictions)
            batch_loss = root_mean_squared_error(y_batch, predictions[:len(y_batch)])
            loss.append(batch_loss)
            print("Validation loss for Theta model:", batch_loss)
    mean_loss = np.mean(loss)
    print("Total Validation  Loss",mean_loss)
    return loss,pred_theta_test
training = len(X_val_tensor)
horizon = len(y_val_tensor)

pred_theta_val = thetamodel_val(validation, val_horizon, val_loader)

# Testing set predictions

def thetamodel_test(train_len, horizon, loader):
    pred_theta_test = []
    loss=[]
    horizon = int(horizon)
    for i, (X_batch, y_batch) in enumerate(loader):
        for j in range(X_batch.shape[0]):
             input_batches= X_batch[j]    
             tm = ThetaModel(endog = input_batches, period=7)
             res = tm.fit()
             predictions = res.forecast(horizon)
             pred_theta_test.extend(predictions)
             batch_loss = root_mean_squared_error(y_batch, predictions[:len(y_batch)])
             loss.append(batch_loss)
             print("Testing loss for Theta model:", batch_loss)
    mean_loss = np.mean(loss)
    print("Total Loss",mean_loss)
    return loss,pred_theta_test

testing = len(X_test_tensor)
test_horizon = len(y_test_tensor)

loss,pred_theta_test = thetamodel_test(testing, test_horizon, test_loader)

# Converting the lists of list into single list inorder to create a dataframe
datetime = [item for sublist in y_datetime_test for item in sublist]
actual = [item for sublist in y_test for item in sublist]
# pred = [item for sublist in pred_theta_test for item in sublist]

df_win_first = pd.DataFrame(list(zip(datetime,actual, pred_theta_test)),
              columns=['datetime',y_cordinate,'predictions'])


### Final plot

# RNN Model
fig, ax = plt.subplots(2,figsize=(12, 6))
ax[0].set_title("Prediction of new Covid-19 cases")
ax[0].plot(data_for_scaling.index, data_for_scaling[y_cordinate], label = 'New covid cases')
ax[0].plot(df_result['datetime'],df_result[y_cordinate], 'r-', label='Values to be predicted')
ax[0].plot(df_result['datetime'],df_result['prediction'], 'b--.', label='RNN model predictions')
ax[0].legend(loc='best')

ax[1].plot(data_for_scaling.index,data_for_scaling[y_cordinate])
ax[1].plot(df_win_first['datetime'],df_win_first[y_cordinate], 'r-', label='Values to be predicted')
ax[1].plot(df_win_first['datetime'],df_win_first['predictions'], 'g--.', label='Theta Model predictions')
#plt.ticklabel_format(style='plain', axis='y')
ax[1].set_xlabel('DateTime')
ax[1].set_ylabel(y_cordinate)
#plt.title("THETA MODEL", fontsize=16)
ax[1].legend(loc='best')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

