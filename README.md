# Time-series

The data used in this project is collected from “European Centre for Disease Prevention and
Control (ECDC)”, an agency of the European Union, in 30 EEA countries. 
Data file contains information about testing volume for COVID-19 by week and country. 
Each row contains the corresponding data for a country and a week.
The file is updated weekly. It provides accurate statistics of weekly testing rates and weekly test positivity based on multiple data sources. 
A large number of weekly univariate time series of confirmed cases are analysed for generating future predictions.

Data Modification:
To begin with, the null values were removed from all the rows and columns of the dataset, because null values may bias the results and reduce the accuracy of the model. It is crucial to eliminate any null values present in the dataset before using it for training any machine learning model.
In the next step, all the values of all countries were merged/grouped with respect to year week column. Because it is difficult to analyse COVID-19 case trends in every country individually, so I decided to work with all the confirmed cases each week.

 Standardization:
Standardizing the features around the mean 0 with a standard deviation of 1 is important when we compare measurements that have different scales. 
A bias can occur if the variables are measured at different scales and don’t contribute equally to the analysis. 
The greater the scale of a variable, the greater the influence it has on any measurement we use. In my data set, the COVID-19 confirmed cases are in a different range, starting from 0 to more than 400000. So, the data was transformed into a comparable scale using standardization. To standardize the data, I take each value and subtract the mean of all values and divide them by the standard deviation. 
The equation is: z = (x − µ) /σ , where, x is the individual value, µ is the mean, and σ is the standard deviation.

Windowing:
Windowing is the process of splitting a sequence of data into overlapping subsequences. This is often used when training machine learning models that take in a sequence of data and output a prediction for the next time step. For example, if you have a time series dataset with 10 data points, you could split it into windows of size 5 with a stride of 2, which would give you the following subsequences: [0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9, 10]

 Splitting the Data:
The specific ratio of the split between the training, validation, and test sets depends on the size of the dataset and the specific requirements of the application. For our forecasting model, I split the data into 60%, 20% and 20% for training, testing and validation respectively. 

RNN Forecasting Model
The designed RNN model is a Recurrent Neural Network implemented in PyTorch, which is designed to process sequential data. It is composed of an RNN layer that processes the input data and a fully connected linear layer that converts the output of the RNN layer to the desired output shape and this model has the option to use dropout to prevent overfitting.
The RNN layer, which is defined using PyTorch and is created with the given parameters of input dim, hidden dim, and layer dim.
forward() method defined on class, is used to pass the input through the RNN layers, and returns the final output of the model.
The Optimization class, which is used to train the RNN model with a specific loss function and optimizer. In the constructor, the class takes in a model, a loss function, and an optimizer as inputs, and initializes them as class variables. It also initializes two empty lists, train losses and val losses, which will be used to store the training and validation losses at each epoch.
The train step method is used to train the model on a single batch of data. It first sets the model to train mode, then makes predictions on the input data using the forward() method of the model. It then calculates the loss using the loss function passed as an argument, and computes the gradients of the model with respect to the loss using loss.backward() method.The train method then loops through a number of epochs, where in each epoch it goes through the entire training data using a train loader and uses the train step function to train the model on each batch of data, appending the loss to the batch losses list. It then computes the mean
of the batch losses as the training loss, and stores it in the train losses list.
It then does the same for validation set, where it uses a val loader to go through the validation data, and computes the mean of the validation losses as the validation loss, and stores it in the val losses list. This process will be repeated for the number of specified epochs and at each iteration, it will print the training loss and validation loss.The evaluate method is used to evaluate the model on the test data. It loops through the test data using a test loader, sets the model to evaluation mode, and makes predictions on the test data. It stores the predictions and the actual values in separate lists, and returns these lists when the method is called.

Theta model(statistical model):

 Identification of Seasonality:
A seasonality test was performed before the implementation of our Theta model. Seasonality is a characteristic of a time series in which the data experiences regular and predictable changes, such as weekly and monthly. Seasonal behaviour is different from cyclic because seasonality is always of a fixed and known period while cyclicity does not have a fixed period. The key for time-series analysis is to understand how the seasonality affects our series, therefore making us produce better forecasts for the future. Our time series data is not necessarily seasonal by itself. 
There are several methods that can be used to identify seasonality in a time series. Some common methods include: 1) Visual inspection: Plotting the time series data and looking for repeating patterns at specific intervals (such as weekly or yearly patterns) can be a useful way to identify seasonality. 2) Autocorrelation function (ACF) plot: The ACF is a statistical measure of the correlation between a time series and a lagged version of itself. An ACF plot can be used to identify periodic patterns in the data, such as seasonality that repeat over a specific time interval, such as weekly or yearly patterns.

Seasonal decomposition: Seasonal decomposition is a statistical method that decomposes the time series data into its trend, seasonality, and residual components. The seasonality component can be used to identify repeating patterns within the data. There are two main types of seasonal decomposition methods: additive decomposition and multiplicative decomposition.
Additive decomposition is based on the assumption that the data is made up of an additive combination of these components, such that the data can be represented as: data = trend + seasonality + residual
Additive decomposition is best suited for time series data with a constant variance, as the residual component will have a constant variance in this case.
Multiplicative decomposition is based on the assumption that the data is made up of a multiplicative combination of these components, such that the data can be represented as: data = trend ∗ seasonality ∗ residual
Multiplicative decomposition is best suited for time series data with a varying variance, as the residual component will have a varying variance in this case.
As the data contains zero or near-zero values and some negative values after scaling, multiplicative decomposition is not appropriate for our time series data set. In that case, I used additive decomposition to identify seasonality. If we look at the seasonal component Figure it can be easily identified that there is no seasonality present.

Building Theta model:

The thetamodel train() function trains the Theta model on the training dataset and returns predictions and the loss. The function performs the following operations: The pred theta test list is used to store the model’s predictions for each batch of data. This allows the function to keep track of all the predictions made by the model as it processes each batch of the data. After processing all the batches, the entire list of predictions can be returned to be used for further evaluation of the model’s performance or for making predictions on new data.
The loss list is used to store the loss for each batch of data. Loss is a measure of how well the model is performing on the data. By keeping track of the loss for each batch, the function can monitor how the model’s performance changes as it processes more data. The average loss over all batches can be calculated and returned, which is a good indicator of how well the model is performing on the overall dataset. A Theta Model object is created, with the input batches being the input dataset and the period being 7 which is the period of the data. Theta model uses the input as endog and the period as exog. Data is collected every 7 day, therefore this value is passed as a exogenous variable to the Theta model, to allow it to capture the periodical patterns in the data and make more accurate predictions.
By calling the tm.f it() method, the Theta Model uses the input data to train its parameters, so that it can make accurate predictions on new data in the future. It first initializes the model’s parameters using the input data. It then optimizes the parameters of the model by minimizing the difference between the model’s predictions and the true values in the input data. The optimization process essentially finds the set of parameters that minimize the difference between the model’s predictions and the true values. Once the optimization process is completed, the model’s parameters are set to the values that minimized the loss, and the model is considered 'trained’ on the input data.
The model then makes predictions using the res.forecast(horizon) method. These predictions are then appended to the pred theta test list.
The batch loss is calculated using root mean squared error method, root mean squared error()
and the result is appended to the loss list. The function thetamodel val() and thetamodel test() are similar to the function thetamodel train() where the input passed to the function is the validation set and testing set respectively, which makes predictions for those datasets respectively.
