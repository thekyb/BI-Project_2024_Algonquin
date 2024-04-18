#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
# Load your dataset
data = pd.read_csv("./ml_scripts/ML_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['date_numeric'] = pd.to_numeric(pd.to_datetime(data['Date']))

key_strings = ['Date_numeric','Workload sum','Scheduled Hours','Planned Absences','Unpaid Absences','Actual Work Hours','EWH by Load','Daily remain','Rough esteam of merged workload','Max Temp(C)','Max Temp Flag','Min Temp(C)','Min Temp Flag','Mean Temp(C)','Mean Temp Flag','Heat Deg Days(C)','Heat Deg Days Flag','Cool Deg Days(C)','Cool Deg Days Flag','Total Rain(mm)','Total Rain Flag','Total Snow(cm)','Total Snow Flag','Total Precip(mm)','Total Precip Flag','Snow on Grnd(cm)','Snow on Grnd Flag','Dir of Max Gust(10s deg)','Dir of Max Gust Flag','Spd of Max Gust(km/h)','Spd of Max Gust Flag','holiday','NATIONAL_HOLIDAY_IND','PROVINCIAL_HOLIDAY_IND','IMPACT_DAY_FLG','DAY_OF_WEEK','WEEK_OF_YEAR','DAY_OF_MONTH','MONTH_NO','YEAR_NO','HolidayFlag','3days_stack_amount'] 

# Select columns containing key strings
X = data[[col for col in data.columns if any(key in col for key in key_strings)]]
# X = pd.get_dummies(X, columns=['day of week'])
minmax = MinMaxScaler()
scaler = StandardScaler()


y = data["Planned Absences"]

# import matplotlib.pyplot as plt

# # Sample data

# # Plot the line chart
# plt.plot(X['date_numeric'], X['Scheduled Hours_NT'], linestyle='-')

# plt.plot(X['date_numeric'], y, linestyle='-')

# # Add labels and title
# plt.title('Line Chart Example')

# # Add grid
# plt.grid(True)

# # Show the plot
# plt.show()
# y = scaler.fit_transform(data["Workload sum"])
print(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(40, 20),
                     activation='relu',
                     solver='adam',
                     max_iter=100,
                     random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)


# # Evaluate the model
train_loss = mean_squared_error(y_train, y_pred_train)
test_loss = mean_squared_error(y_test, y_pred_test)
print("Train Loss(mse):", train_loss)
print("Test Loss(mse):", test_loss)

rmse = np.sqrt(test_loss)
print("test RMSE = ", rmse)
# # %%

# %%
class BankAccount:

    def __init__(self, account_number, balance):

        self.account_number = account_number

        self.balance = balance


    def deposit(self, amount):

        self.balance += amount


    def withdraw(self, amount):

        if amount < self.balance:

            self.balance -= amount

        else:

            return "Insufficient funds"


    def get_balance(self):

        return self.balance


account = BankAccount("123456", 1000)

account.deposit(500)

account.withdraw(200) 
# %%
account.balance
# %%

person = {'name': 'Alex', 'age': 28, 'city': 'San Francisco', 'salary': 90000}

b = 'gender' in person.keys()
print (b)
# %%
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_squared_error

# # Load your dataset
# data = pd.read_csv("your_dataset.csv")

# # Separate features (X) and target variable (y)
# X = data.drop(columns=["Workload sum"])
# y = data["Workload sum"]

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Initialize and train the MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(64, 32),
                     activation='relu',
                     solver='adam',
                     max_iter=100,
                     random_state=42)
model.fit(X_train_scaled, y_train)

# # Predictions
# y_pred_train = model.predict(X_train_scaled)
# y_pred_test = model.predict(X_test_scaled)

# # Evaluate the model
# train_loss = mean_squared_error(y_train, y_pred_train)
# test_loss = mean_squared_error(y_test, y_pred_test)
# print("Train Loss:", train_loss)
# print("Test Loss:", test_loss)

