# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("ml_logistic.csv")

data['Date'] = pd.to_datetime(data['Date'])

# Extract month and year from the date
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year
data['day'] = data['Date'].dt.day
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame

# Calculate correlation coefficients
corr_matrix = data.corr()
# print (data.head)

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)


# %%
# Define features (X) and target variable (y)
X = data.drop(columns=["Workload sum"])  # Assuming "Workload sum" is the target variable
y = data["Workload sum"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))

# Optionally, you can also tune hyperparameters using GridSearchCV or RandomizedSearchCV
#%%
