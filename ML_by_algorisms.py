#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

#%%
# Load the dataset
data = pd.read_csv('./Data/CanadaPost_WorkLoad.csv')
#%%
#
# Drop any rows with missing values
data['Date'] = pd.to_datetime(data['Date'])
data['Day_of_Week'] = data['Date'].dt.dayofweek
# Perform one-hot encoding for 'Day_of_Week'
data = pd.get_dummies(data, columns=['Day_of_Week'], prefix='Day')

sns.stripplot(x='day', y='tip', data= data )
#%%
data = data.dropna()
correlation_matrix = data.corr()

# Display correlation matrix
print(correlation_matrix)
data.head
#%%
#
# Define features and target
X = data.drop(['Date', 'Workload sum'], axis=1)
# X = int(data['Day_of_Week'])

y = data['Workload sum']
#%%
X.head
#%%
y.head
#
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
#
# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
#%%
#
# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
#%%
#
# Ensemble (Voting Classifier)
ensemble_model = VotingClassifier(estimators=[('dt', dt_model), ('rf', rf_model)], voting='hard')
ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_pred))
#%%
#
# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
#%%
#
# Support Vector Machines
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
#%%
#
# Neural Network
nn_model = MLPClassifier(random_state=42)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
print("Neural Network Accuracy:", accuracy_score(y_test, nn_pred))
#%%
#