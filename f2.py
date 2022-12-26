# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 2022
@author: Abhilash Adya, Belaul Emon Hasan

Note: 1) Please assign the path where you have stored the dataset(.csv file) in 'path' variable below.
      2) One can see the graph of any column like 'new_cases' and 'population' vs year_week just by writing down 
         the name of the column in variable 'y_co'. Here, we have used "tests_done".
"""




# importing necessary libraries/functions/classes.

import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import datetime
import time
#from sklearn.model_selection import train_test_split

#from statsmodels.tsa.forecasting.theta import ThetaModel
import torch
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch.optim as optim

# Assigning Variables

y_co = "tests_done"
x_co = 'year_week'
path = os.path.join(os.getcwd(),"Original_data.csv")
win_size = 2
warnings.filterwarnings('ignore')

# Reading given data

data = pd.read_csv(path)


# Checking for null datasets

data[y_co].isnull().values.any()
data[x_co].isnull().values.any()

# Removing the null values from data

data = data.dropna(axis=0, subset=[x_co, y_co])
#print(data[y_co].isnull().values.any())


# Using groupby function to sum the data for all the given countries with respect to year_week column.

var = dict(data.groupby(x_co)[y_co].sum())
#print(var)

# Converting obtained dictionary into dataframe

new_data = pd.DataFrame(var.items(), columns=[x_co, y_co])

# Converting string values of time(here x_co) into timestamps

time = new_data['year_week'].tolist()
time_data = []
for element in time:
 r = datetime.datetime.strptime(element + '-1', "%Y-W%W-%w")
 time_data.append(r)


# Standardization: 0 mean and 1 standard deviation

df_new = (new_data-new_data.mean())/new_data.std()
#print(df_new)

y_scaled = y_co
y_scaled = df_new[df_new.columns[0]].values.tolist()

#print(y_scaled)
# Putting the obtained values in a dataframe, because it is easy to plot a dataframe

df = pd.DataFrame(list(zip(time_data, y_scaled)),
              columns=['time_data','y_scaled'])


df['time_data'] = pd.to_datetime(df['time_data'])

x_scaled = df['time_data'].tolist()
# =============================================================================
# x_array = np.array(x_scaled)
# y_array = np.array(y_scaled)
# =============================================================================

def windowing(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

#windowed_data = create_inout_sequences(df, win_size)
#X_tr,y_val = windowing()


# =============================================================================
# X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(
#      x_scaled, y_scaled, test_size=0.2)
# X_train, X_val, y_train, y_val = train_test_split(
#      X_train_and_val, y_train_and_val, test_size=0.25)
# print(y_train)
# =============================================================================

training = y_scaled[:87]
X_train = x_scaled[:87]
train_data = torch.FloatTensor(training)

validation = y_scaled[87:116]
X_val = x_scaled[87:116]
val_data = torch.FloatTensor(validation)

testing = y_scaled[116:]
X_test = x_scaled[116:]
test_data = torch.FloatTensor(testing)

train_window = windowing(train_data, win_size)
val_window = windowing(val_data,win_size)
test_window = windowing(test_data,win_size)



train_loader = DataLoader(train_window,shuffle=False,batch_size=5,drop_last=True)
val_loader = DataLoader(train_window,shuffle=False,batch_size=5,drop_last=True)
test_loader = DataLoader(train_window,shuffle=False,batch_size=5,drop_last=True)

# Modeling
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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

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
        model_path = f'models/{self.model}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)
        
    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values
    
input_dim = len(X_train)
output_dim = 1
hidden_dim = 5
layer_dim = 3
batch_size = 5
dropout = 0.2
n_epochs = 5
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
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

predictions, values = opt.evaluate(test_loader, batch_size=5, n_features=input_dim)


# Plotting data

plt.figure(figsize=(12,6))
ax = df.plot(x="time_data", y=["y_scaled"],
        kind="line",)
plt.title("COVID-19 DATA FOR EUROPE")
plt.ylabel(y_co,fontweight='bold')
plt.xlabel(x_co,fontweight='bold')
#ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.legend()
plt.show()

