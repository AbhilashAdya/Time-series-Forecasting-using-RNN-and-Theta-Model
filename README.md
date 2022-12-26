# Time-series

The task is to forecast covid-19 tests-done/new-cases based on weekly data for 2020,2021,20222 for europe.
Restriction: None other than Pytorch library needs to be used for modeling. So, keras and Tensor flow are out of options and one needs to convert the data into tensors after pre-processing.
Preprocessing:
The dataset had a lot of null values so initially, null values were checked and removed using .isnull() and .dropna
Now, the given data consists of weekly cases(tests-done/new-cases) for different countries of europe but we need to consider Europe as a whole rather than individual country's data. So, groupby.sum() was used to add the data and group it according to the year-week column and added to the dictionary.
The year-week column is apparently just a string and not in date-time format. So, I converted it into time-stamps(YY-MM-DD) and added it to the data frame(new_data along with new-cases/tests-done.
Now, when one sees the data one can sense that maximum value of new-cases/tests done goes upto or above 40K. This large value has the ability to dominate the forecasting and create bias. Hence, it needs to be standardized/scaled so that the data is equally distributed. In standardization, one needs to bring the mean of the data to 0 and standard deviation to 1. 
Note: The time-stamps are not float() values so the o/p will show NA in year_weak column of the created dataframe(df_new)
After standardization, I took the scaled values(y_scaled) in a separate list and created a new dataframe. Then, I took the time-stamps into a list as well.

Ps: I know, lot of dataframes and lists have been created unnecessarily but that was the need of the situation as I had to plot it and dataframe was my best option to get adjusted x-axis. I will clean the code later on.

Windowing: In order to evaluate the data week by week, one needs to break the data i.e., if there are n weeks then n windows should be obtained with respective number of cases or roll a single window n time.
Next, I defined a function named windowing. The function will accept the raw input data and will return a list of tuples. In each tuple, the first element will contain list of win_size items corresponding to the number of tests-done/newcases inn the timeframe between two weeks, the second tuple element will contain one item i.e. the number of cases in the win_size+1st week.
Then the data was split into training(60%), validation(20%) and testing(20%). This was again another specified task just like Pytorch.
Then, the data was converted into tensor for loading into the Dataloader(Pytorch)
