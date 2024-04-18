#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Load your dataset
data = pd.read_csv("./ml_scripts/ML_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
reference_date = pd.to_datetime('2018-01-01')  # You can choose any reference date you like
data['Date_numeric'] = (data['Date'] - reference_date).dt.days
#%% data cleaning
data.isna

#%%
# Separate features (X) and target variable (y)
X = data.drop(columns=["Date","Workload sum"])
y = data["Workload sum"]



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(64, 32),
                     activation='relu',
                     solver='adam',
                     max_iter=100,
                     random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Evaluate the model
train_loss = mean_squared_error(y_train, y_pred_train)
test_loss = mean_squared_error(y_test, y_pred_test)
print("Train Loss:", train_loss)
print("Test Loss:", test_loss)


# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, clf_model.predict_proba(X_test_scaled)[:, 1])
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, clf_model.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Example: Regression Metrics
X_reg = data.drop(columns=["y"])
y_reg = data["y"]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Standardize features
scaler_reg = StandardScaler()
X_train_scaled_reg = scaler_reg.fit_transform(X_train_reg)
X_test_scaled_reg = scaler_reg.transform(X_test_reg)

# Train a regressor (Random Forest)
reg_model = RandomForestRegressor()  # Example: Random Forest
reg_model.fit(X_train_scaled_reg, y_train_reg)

# Predictions
y_pred_train_reg = reg_model.predict(X_train_scaled_reg)
y_pred_test_reg = reg_model.predict(X_test_scaled_reg)

# Calculate regression metrics
mae = mean_absolute_error(y_test_reg, y_pred_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_test_reg)
r2 = r2_score(y_test_reg, y_pred_test_reg)

# Plotting Learning Curves (Optional)
# from sklearn.model_selection import learning_curve
# train_sizes, train_scores, valid_scores = learning_curve(reg_model, X_reg, y_reg, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5)
# plt.plot(train_sizes, np.mean(train_scores, axis=1), 'r-', label='Train Score')
# plt.plot(train_sizes, np.mean(valid_scores, axis=1), 'b-', label='Validation Score')
# plt.xlabel('Training examples')
# plt.ylabel('Score')
# plt.legend()
# plt.show()

# Print Metrics
print("Classification Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC:", roc_auc)
print("Confusion Matrix:")
print(conf_matrix)

print("\nRegression Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
# %%
