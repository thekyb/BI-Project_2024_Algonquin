#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
# Load your dataset
data = pd.read_csv("./ml_scripts/ML_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['date_numeric'] = pd.to_numeric(pd.to_datetime(data['Date']))
# key_strings = ['date_numeric', 'Scheduled Hours_ST']

key_strings = ['date_numeric', 'Actual Work Hours', 'Scheduled Hours']

# Select columns containing key strings
X = data[[col for col in data.columns if any(key in col for key in key_strings)]]
minmax = MinMaxScaler()
scaler = StandardScaler()


y = data["Workload sum"]

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
model = MLPRegressor(hidden_layer_sizes=(400, 20),
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
print(rmse)
# # %%


import matplotlib.pyplot as plt

plt.plot(y_pred_test, label='Predicted')  # Plot predicted values
plt.plot(y_test, label='Actual')          # Plot actual values
plt.title('Neural Network Output vs Actual')
plt.xlabel('Sample')
plt.ylabel('Output')
plt.legend()  # Add a legend to distinguish between predicted and actual values
plt.show()

# %%
