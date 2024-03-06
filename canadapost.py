
#%%
!pip3 install openpyxl
# %%
import pandas as pd
# Load the Excel file into a DataFrame
df = pd.read_excel(r'CPC-Algonquin College - Depot Dashboard - Data v2 - 2019-24.xlsx')


# %%
# Display the first few rows of the DataFrame
print(df.head())


# %%
# Get information about the DataFrame
print(df.info())


# %%
# Summary statistics for numerical columns
print(df.describe())


# %%
# Check dimensions of the dataset
print("Dimensions of the dataset:", df.shape)


# %%
# Summary statistics for numerical columns
print("Summary statistics for numerical columns:")
print(df.describe())

# Analyze the distribution of numerical variables
import matplotlib.pyplot as plt
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols].hist(bins=20, figsize=(12, 8))
plt.show()


# %%
# Check the unique values in the 'CALENDAR_DATE' column
print("Unique values in the 'CALENDAR_DATE' column:")
print(df['CALENDAR_DATE'].unique())


# %%
# Remove rows with unexpected values in 'CALENDAR_DATE' column
df = df[df['CALENDAR_DATE'] != 'Result Status:']

# Now, try converting 'CALENDAR_DATE' to datetime format again
df['CALENDAR_DATE'] = pd.to_datetime(df['CALENDAR_DATE'])

# Analyze trends over time
plt.figure(figsize=(12, 8))
plt.plot(df['CALENDAR_DATE'], df['ACTUAL_VOLUME'])
plt.title('Trends of ACTUAL_VOLUME over time')
plt.xlabel('Date')
plt.ylabel('ACTUAL_VOLUME')
plt.show()


# %%
#Box plots for numerical variables
df[numerical_cols].boxplot(figsize=(12, 8))
plt.show()


# %%
# Bar plots for categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].value_counts().plot(kind='bar', figsize=(8, 6), title=col)
    plt.show()


# %%
correlation_matrix = df.corr()
print(correlation_matrix)


# %%
import seaborn as sns

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# %%
# Examine the frequency and distribution of holiday indicators
holiday_freq = df['NATIONAL_HOLIDAY_IND'].value_counts()
print("Frequency of National Holiday Indicators:")
print(holiday_freq)

holiday_dist = df.groupby('NATIONAL_HOLIDAY_IND').size() / len(df)
print("\nDistribution of National Holiday Indicators:")
print(holiday_dist)

# %%
# Visualize the distribution of holiday indicators
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='NATIONAL_HOLIDAY_IND')
plt.title('Distribution of National Holiday Indicators')
plt.xlabel('National Holiday Indicator')
plt.ylabel('Count')
plt.show()

# %%
# Check for null values in all columns
print("Null values in each column:")
print(df.isnull().sum())


# %%
# Select holiday indicator columns
holiday_indicator_cols = ['NATIONAL_HOLIDAY_IND', 'PROVINCIAL_HOLIDAY_IND', 'IMPACT_DAY_FLG']

# Create box plots for holiday indicator columns
plt.figure(figsize=(12, 8))
df[holiday_indicator_cols].boxplot()
plt.title('Box plot of Holiday Indicator Columns')
plt.ylabel('Holiday Indicator Value')
plt.xlabel('Holiday Indicator')
plt.xticks(rotation=45)
plt.show()


# %%
# Remove the "Province code" column
df_cleaned = df.drop(columns=['SITE_PROVINCE_CODE'])

# %%
df_cleaned

# %%
# Importing necessary libraries
import matplotlib.pyplot as plt

# Plotting against each holiday indicator
plt.figure(figsize=(12, 8))

# Scatter plot for NATIONAL_HOLIDAY_IND vs. ACTUAL_VOLUME
plt.subplot(1, 3, 1)
plt.scatter(df['NATIONAL_HOLIDAY_IND'], df['ACTUAL_VOLUME'])
plt.title('NATIONAL_HOLIDAY_IND vs. ACTUAL_VOLUME')
plt.xlabel('NATIONAL_HOLIDAY_IND')
plt.ylabel('ACTUAL_VOLUME')

# Scatter plot for PROVINCIAL_HOLIDAY_IND vs. ACTUAL_VOLUME
plt.subplot(1, 3, 2)
plt.scatter(df['PROVINCIAL_HOLIDAY_IND'], df['ACTUAL_VOLUME'])
plt.title('PROVINCIAL_HOLIDAY_IND vs. ACTUAL_VOLUME')
plt.xlabel('PROVINCIAL_HOLIDAY_IND')
plt.ylabel('ACTUAL_VOLUME')

# Scatter plot for IMPACT_DAY_FLG vs. ACTUAL_VOLUME
plt.subplot(1, 3, 3)
plt.scatter(df['IMPACT_DAY_FLG'], df['ACTUAL_VOLUME'])
plt.title('IMPACT_DAY_FLG vs. ACTUAL_VOLUME')
plt.xlabel('IMPACT_DAY_FLG')
plt.ylabel('ACTUAL_VOLUME')

plt.tight_layout()
plt.show()


# %%



