# COVID-19 Data Analysis and Prediction

## Introduction
This repository contains code for the analysis and prediction of COVID-19 data using Python. It includes data preprocessing, time series analysis, and prediction using both Recurrent Neural Networks (RNN) and the Theta model.

## Requirements
Before running the code, make sure you have the following dependencies installed:

## Python 3.x
Libraries: pandas, matplotlib, numpy, scikit-learn, torch, statsmodels
You can install the required libraries using pip:

## Copy code
pip install pandas matplotlib numpy scikit-learn torch statsmodels

## Usage
Data Preparation:

Ensure you have the COVID-19 dataset in CSV format stored in the path variable in the code. Make sure to assign the correct file path.
Specify the column names for the x and y coordinates (y_cordinate and x_cordinate) based on the data you want to analyze and predict.
Training RNN Model:

The code includes an RNN model for time series prediction. You can adjust hyperparameters such as win_size, batch_size, n_epochs, and learning_rate to fine-tune the model.
The trained model's performance is evaluated and plotted.
Theta Model:

The code also implements the Theta model for time series forecasting. It includes training, validation, and testing.
The results of the Theta model are plotted alongside the actual data.
Plotting Results:

The final plots show the actual COVID-19 data, RNN model predictions, and Theta model predictions.

## Error Metrics:

The code calculates the Root Mean Squared Error (RMSE) to assess the accuracy of predictions.

## Author
Abhilash Adya
