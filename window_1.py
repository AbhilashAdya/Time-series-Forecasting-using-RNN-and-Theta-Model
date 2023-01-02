import main
import pandas as pd
import numpy as np
from datetime import datetime

df_for_windowing = pd.DataFrame({'year_week': main.scaled_data.index, 'tests_done': main.scaled_data.values})
size_of_df = df_for_windowing.size

for element in df_for_windowing['year_week']:
    element = datetime.strptime(element + '-1', "%Y-W%W-%w")

# print(type(df_for_windowing['year_week']))
df_for_windowing.index = df_for_windowing['year_week']
df_for_windowing.drop(columns=['year_week'], inplace=True)
# print(df_for_windowing.head())
# print(df_for_windowing.index)

# print(df_for_windowing.head())

l = 2
m = 1
n = 2
# print(df_for_windowing["tests_done"].head())
np_data = np.array(df_for_windowing)
N = np_data.shape[0]
k = N - (l+m+n)

in_slice = np.array([range(i, i + l) for i in range(k)])
op_slice = np.array([range(i + l + m, i + l + m + n) for i in range(k)])

in_data = np_data[in_slice, :]
print(in_data)
print("op sliced data after this")
out_data = np_data[op_slice, :]
print(out_data)
