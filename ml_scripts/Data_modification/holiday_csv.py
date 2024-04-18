# %% 
import pandas as pd

# Define the holiday list
holiday_list = [
    '2019-01-01', '2019-02-18', '2019-04-19', '2019-04-22', '2019-05-20',
    '2019-07-01', '2019-08-05', '2019-09-02', '2019-10-14', '2019-11-11',
    '2019-12-25', '2019-12-26', '2021-01-01', '2021-02-15', '2021-04-02',
    '2021-04-05', '2021-05-24', '2021-07-01', '2021-08-02', '2021-09-06',
    '2021-09-30', '2021-10-11', '2021-11-11', '2021-12-25', '2021-12-26',
    '2023-01-01', '2023-02-21', '2023-04-15', '2023-04-18', '2023-05-23',
    '2023-07-01', '2023-08-01', '2023-09-05', '2023-09-19', '2023-09-30',
    '2023-10-10', '2023-11-11', '2023-12-25', '2023-12-26', '2024-01-01',
    '2024-02-20', '2024-04-07', '2024-04-10', '2024-05-22', '2024-07-01',
    '2024-08-07', '2024-09-04', '2024-09-30', '2024-10-09', '2024-11-11',
    '2024-12-25', '2024-12-26'
]

# Generate date range from 2019-01-01 to 2024-12-31
start_date = '2019-01-01'
end_date = '2024-12-31'
date_range = pd.date_range(start=start_date, end=end_date)

# Create DataFrame with consecutive dates
df = pd.DataFrame({'date': date_range})

# Convert 'date' column to string format
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# Add 'holiday' column
df['holiday'] = df['date'].isin(holiday_list).astype(int)

# Save DataFrame to CSV
df.to_csv('consecutive_dates_with_holidays.csv', index=False)


# %%
