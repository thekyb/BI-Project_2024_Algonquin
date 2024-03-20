# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
def convert_to_numeric(column):
    return pd.to_numeric(column, errors='coerce').fillna(0).astype(int)

# %%
df = pd.read_csv('./data/canadapost_workload_covid_removed.csv')
df.head
df.columns
# %% Standardize the column
# dataset["Unplanned Absences (Sum)"]

# dataset["Unplanned Absences (Sum)"]

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('./data/canadapost_workload_covid_removed.csv')

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
st = StandardScaler

print (numerical_columns)
# Assuming 'column_to_standardize' is the column you want to standardize

column_to_standardize= "Spd of Max Gust(km/h)"
# a = True
# %%
df[column_to_standardize + '_standardized'] = st.fit_transform(df[[column_to_standardize]])
# %%
for column in numerical_columns:
    # if a :
        # print(column)
        # a = False
    
    df[column+"_ST"] = st.fit_transform(df[[column]])

# Save the modified DataFrame back to a CSV file
df.to_csv('modified_file.csv', index=False, )




# %% 
dataset.iloc[:, 9:-1] = dataset.iloc[:, 9:-1].apply(convert_to_numeric)

dataset


# %%
dataset.to_csv('pop2022_fillnumeric.csv')

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('./data/canadapost_workload_covid_removed.csv')

# Assuming 'column_to_standardize' is the column you want to standardize
column_to_standardize = 'Unplanned Absences'

# Create a StandardScaler instance
scaler = StandardScaler()

# Standardize the selected column
df[column_to_standardize + '_standardized'] = scaler.fit_transform(df[[column_to_standardize]])

# Save the modified DataFrame back to a CSV file
df.to_csv('modified_file.csv', index=False)
# %%
column_to_standardize = 'Unplanned Absences'


Date', 'Workload sum', 'Scheduled Hours', 'Planned Absences',
       'Unplanned Absences', 'Unpaid Absences', 'Actual Work Hours',
       'EWH by Load', 'Daily remain', 'Rough esteam of merged workload',
       'Max Temp(C)', 'Max Temp Flag', 'Min Temp(C)', 'Min Temp Flag',
       'Mean Temp(C)', 'Mean Temp Flag', 'Heat Deg Days(C)',
       'Heat Deg Days Flag', 'Cool Deg Days(C)', 'Cool Deg Days Flag',
       'Total Rain(mm)', 'Total Rain Flag', 'Total Snow(cm)',
       'Total Snow Flag', 'Total Precip(mm)', 'Total Precip Flag',
       'Snow on Grnd(cm)', 'Snow on Grnd Flag', 'Dir of Max Gust(10s deg)',
       'Dir of Max Gust Flag', 'Spd of Max Gust(km/h)',
       'Spd of Max Gust Flag'],
      dtype='object'

column_to_standardize = 'Unplanned Absences'
df[column_to_standardize + '_standardized'] = scaler.fit_transform(df[[column_to_standardize]])

