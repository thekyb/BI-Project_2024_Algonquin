# %%
import os

current_directory = os.getcwd()
print("Current directory:", current_directory)
# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%


# %%
df = pd.read_csv('./data/sum7_days.csv')
# df = pd.read_csv('./data/canadapost_workload_covid_removed.csv')
df.head
# %% Standardize the column
# dataset["Unplanned Absences (Sum)"]

# dataset["Unplanned Absences (Sum)"]

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_season(date):
    month = date.month
    if month in range(3, 6):
        return 1
    elif month in range(6, 9):
        return 2
    elif month in range(9, 12):
        return 3
    else:
        return 4
    
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('./data/ML_data_ST_N.csv')
key_strings = ['Scheduled Hours', 'Scheduled Hours_ST', 'Rough esteam of merged workload_ST', 'Planned Absences','Unplanned Absences','Daily remain','Max Temp(C)','Min Temp(C)', 'Total Precip(mm)',  'Total Rain(mm)', 'Total Snow(cm)','holiday']
df["Total Snow(cm)"]
# Select columns containing key strings
dfn = df[[col for col in df.columns if any(key in col for key in key_strings)]]
df['Date'] = pd.to_datetime(df['Date'])
#%%
# Select only numerical columns for standardization
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
key_strings = ['Scheduled Hours', 'Daily remain','Max Temp(C)','Min Temp(C)', 'Total Precip(mm)',  'Total Rain(mm)', 'Total Snow(cm)']
#
# Create a StandardScaler instance
scaler = StandardScaler()
minmax = MinMaxScaler()
# Standardize each numerical column using a for loop
# for column in key_strings:
#     df[column+'_ST'] = scaler.fit_transform(df[[column]])
for column in key_strings:
    dfn[column+'_ST'] = minmax.fit_transform(df[[column]])


dfn['day'] = df['Date'].dt.day
dfn['mnth'] = df['Date'].dt.month
dfn['year'] = df['Date'].dt.year
dfn['day_of_week'] = df['Date'].dt.dayofweek
dfn['season'] = df['Date'].apply(get_season)
dfn['unplanned Absence ratio'] = df['Date'].apply(get_season)



dfn.to_csv('modified_file.csv', index=False)
#%%



# Save the modified DataFrame back to a CSV file
df.to_csv('modified_file.csv', index=False)

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
# column_to_standardize = 'Unplanned Absences'


column_to_standardize = 'Date'
column_to_standardize = 'Workload sum'
column_to_standardize = 'Scheduled Hours'
column_to_standardize = 'Planned Absences'
# column_to_standardize = 'Unplanned Absences'
# column_to_standardize = 'Unpaid Absences'
# column_to_standardize = 'Actual Work Hours'
# column_to_standardize = 'EWH by Load'
# column_to_standardize = 'Daily remain'
# column_to_standardize = 'Rough esteam of merged workload'
# column_to_standardize = 'Max Temp(C)'
# column_to_standardize = 'Max Temp Flag'
# column_to_standardize = 'Min Temp(C)'
# column_to_standardize = 'Min Temp Flag'
# column_to_standardize = 'Mean Temp(C)'
# column_to_standardize = 'Mean Temp Flag'
# column_to_standardize = 'Heat Deg Days(C)'
# column_to_standardize = 'Heat Deg Days Flag'
# column_to_standardize = 'Cool Deg Days(C)'
# column_to_standardize = 'Cool Deg Days Flag'
# column_to_standardize = 'Total Rain(mm)'
# column_to_standardize = 'Total Rain Flag'
# column_to_standardize = 'Total Snow(cm)'
# column_to_standardize = 'Total Snow Flag'
# column_to_standardize = 'Total Precip(mm)'
# column_to_standardize = 'Total Precip Flag'
# column_to_standardize = 'Snow on Grnd(cm)'
# column_to_standardize = 'Snow on Grnd Flag'
# column_to_standardize = 'Dir of Max Gust(10s deg)'
# column_to_standardize = 'Dir of Max Gust Flag'
# column_to_standardize = 'Spd of Max Gust(km/h)'
# column_to_standardize = 'Spd of Max Gust Flag'

column_to_standardize = 'Unplanned Absences'
df[column_to_standardize + '_standardized'] = scaler.fit_transform(df[[column_to_standardize]])


# %%
import pandas as pd

data = pd.read_csv("./data/canadapost_workload_covid_removed.csv")

# Sum every 7 consecutive rows
df = data.groupby(data.index // 30).sum()

df['Date'] = data['Date'].groupby(data['Date'].index // 30).first()
df.to_csv('./sum30_days.csv', index=False)
# %%
import pandas as pd

# %%

## merging data through month
import pandas as pd
from glob import glob

# Assuming all CSV files are in the same directory
# You can adjust the path accordingly
data = pd.read_csv("./data/canadapost_workload_covid_removed.csv")

# List to hold dataframes for each month
# Read CSV file into a dataframe

# Assuming your CSV has a column named 'date' containing dates
# You may need to adjust the column name
temp = pd.to_datetime(data['Date'])
print(data.head)
# Extract month and year from the date
data['DatebyMon'] = temp.dt.month.to_string() + '\/'+ temp.dt.year.to_string()
print(data.head)
#
data.to_csv('./sum30_days.csv', index=False)

# Append dataframe to the list
# df['Date'] = pd.to_numeric(df['Date'])


# Concatenate all dataframes into one
# df.head()
# Sum every 7 consecutive rows
# df.set_index('DatebyMon')
# df = data.groupby('DatebyMon').sum()

# df.to_csv('./sum30_days.csv', index=False)

# %%
## Data cleaning: make holiday flag
import csv
from datetime import datetime, timedelta

def read_holidays_from_csv(csv_file):
    holidays = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            holiday_date = datetime.strptime(row['Date'], '%m/%d/%Y').date()
            holiday_status = int(row['holiday'])
            holidays[holiday_date] = holiday_status
    return holidays

def write_holidays_with_flags(input_csv, output_csv):
    holidays = read_holidays_from_csv(input_csv)
    
    with open(input_csv, 'r') as file, open(output_csv, 'w', newline='') as output:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames + ['HolidayFlag']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            date = datetime.strptime(row['Date'], '%m/%d/%Y').date()
            previous_day = date - timedelta(days=1)
            next_day = date + timedelta(days=1)
            # weekend = 1 if row['DAY_OF_WEEK'] == '1' or row['DAY_OF_WEEK'] == '7' else 0
            # if weekend: print("Day_of_week", row['DAY_OF_WEEK']) 

            previous_day_holiday = holidays.get(previous_day, 0)
            current_day_holiday = holidays.get(date, 0)
            next_day_holiday = holidays.get(next_day, 0)
            
            # Merge the flags for previous day, current day, and next day into a single flag
            holiday_flag = 1 if previous_day_holiday or next_day_holiday else 0
            
            row['HolidayFlag'] = holiday_flag
            
            writer.writerow(row)

input_csv_file = "C:\\repo\\canadapost_datamodification\\ML_data_ST_N.csv"# Change this to your CSV file name
output_csv_file = 'holidays_with_combined_flag.csv'  # Change this to desired output CSV file name

write_holidays_with_flags(input_csv_file, output_csv_file)



# %%
import csv
from datetime import datetime, timedelta

def read_holidays_from_csv(csv_file):
    holidays = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            holiday_date = datetime.strptime(row['Date'], '%m/%d/%Y').date()
            holiday_status = int(row['holiday'])
            holidays[holiday_date] = holiday_status
    return holidays

def write_holidays_with_flags(input_csv, output_csv):
    holidays = read_holidays_from_csv(input_csv)
    
    with open(input_csv, 'r') as file, open(output_csv, 'w', newline='') as output:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames + ['HolidayFlag']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            date = datetime.strptime(row['Date'], '%m/%d/%Y').date()
            previous_day = date - timedelta(days=1)
            next_day = date + timedelta(days=1)
            
            previous_day_holiday = holidays.get(previous_day, 0)
            current_day_holiday = holidays.get(date, 0)
            next_day_holiday = holidays.get(next_day, 0)
            

            # Merge the flags for previous day, current day, and next day into a single flag
            holiday_flag = 1 if previous_day_holiday or current_day_holiday or next_day_holiday else 0
            
            row['HolidayFlag'] = holiday_flag
            
            writer.writerow(row)

input_csv_file = 'holidays_added.csv'  # Change this to your CSV file name
output_csv_file = 'holidays_with_combined_flag.csv'  # Change this to desired output CSV file name

write_holidays_with_flags(input_csv_file, output_csv_file)



# %% add 3days snowstack
import pandas as pd

# Read the CSV file
df = pd.read_csv('holidays_with_combined_flag.csv')

# Assuming 'snow_fall' is the column containing snowfall data
snowfall_column = 'Total Snow(cm)'

# Calculate the 3-day stack amount
df['3days_stack_amount'] = df[snowfall_column].rolling(window=3).sum()

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_file.csv', index=False)

# %% make one column nomalize and standardize
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Read the CSV file
df = pd.read_csv('updated_file.csv')

# Assuming 'snow_fall' is the column containing snowfall data
snowfall_column = '3days_stack_amount'

scaler = StandardScaler()
minmax = MinMaxScaler()

# Min-max normalization
df[snowfall_column+'_NT'] = minmax.fit_transform(df[[snowfall_column]])
df[snowfall_column+'_ST'] = scaler.fit_transform(df[[snowfall_column]])

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_file2.csv', index=False)

# %%


# %%
import csv
from datetime import datetime, timedelta

def read_holidays_from_csv(csv_file):
    holidays = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            holiday_date = datetime.strptime(row['Date'], '%m/%d/%Y').date()
            holiday_status = int(row['holiday'])
            holidays[holiday_date] = holiday_status
    return holidays

def write_holidays_with_flags(input_csv, output_csv):
    holidays = read_holidays_from_csv(input_csv)
    
    with open(input_csv, 'r') as file, open(output_csv, 'w', newline='') as output:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames + ['Holiday_with_weekend']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            date = datetime.strptime(row['Date'], '%m/%d/%Y').date()
            weekend = 1 if row['DAY_OF_WEEK'] == '1' or row['DAY_OF_WEEK'] == '7' else 0

            # Merge the flags for previous day, current day, and next day into a single flag
            holiday_encluding_weekend = 1 if row['holiday'] == '1' or weekend else 0
            
            row['Holiday_with_weekend'] = holiday_encluding_weekend 
            
            writer.writerow(row)

input_csv_file = 'holidays_added.csv'  # Change this to your CSV file name
output_csv_file = 'holidays_with_combined_flag.csv'  # Change this to desired output CSV file name

write_holidays_with_flags(input_csv_file, output_csv_file)


# %%
