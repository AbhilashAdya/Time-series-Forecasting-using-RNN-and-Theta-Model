import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

original_data = pd.read_csv("Original_data.csv", usecols=['year_week', 'tests_done'])

df_without_null = original_data.dropna()
df_without_null = original_data.dropna().reset_index(drop=True)

grouped_data = df_without_null.groupby(["year_week"])["tests_done"].sum()

scaled_data = (grouped_data - grouped_data.mean()) / grouped_data.std()


plt.figure(figsize=(12, 6))
plt.title("COVID-19 DATA FOR EUROPE")
plt.ylabel('tests_done', fontweight='bold')
plt.xlabel('year_week', fontweight='bold')
scaled_data.plot()
plt.tight_layout()
plt.legend()
plt.show()

