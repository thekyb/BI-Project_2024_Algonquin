
# %%
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

data = pd.read_csv("./ml_scripts/ML_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['date_numeric'] = pd.to_numeric(pd.to_datetime(data['Date']))

# key_strings = ['date_numeric', 'Scheduled Hours', 'Workload sum', 'Rough esteam of merged workload', 'Total Rain(mm)', 'Total Snow(cm)']

# key_strings = ['Date_numeric','Workload sum','Scheduled Hours','Actual Work Hours','EWH by Load','Daily remain','Rough esteam of merged workload','Max Temp(C)','Max Temp Flag','Min Temp(C)','Min Temp Flag','Mean Temp(C)','Mean Temp Flag','Heat Deg Days(C)','Heat Deg Days Flag','Cool Deg Days(C)','Cool Deg Days Flag','Total Rain(mm)','Total Rain Flag','Total Snow(cm)','Total Snow Flag','Total Precip(mm)','Total Precip Flag','Snow on Grnd(cm)','Snow on Grnd Flag','Dir of Max Gust(10s deg)','Dir of Max Gust Flag','Spd of Max Gust(km/h)','Spd of Max Gust Flag','holiday','NATIONAL_HOLIDAY_IND','PROVINCIAL_HOLIDAY_IND','IMPACT_DAY_FLG','DAY_OF_WEEK','WEEK_OF_YEAR','DAY_OF_MONTH','MONTH_NO','YEAR_NO','HolidayFlag','3days_stack_amount'] 
key_strings = ['Date_numeric','Workload sum','Scheduled Hours','Planned Absences','Unplanned Absences','Unpaid Absences','Actual Work Hours','EWH by Load','Daily remain','Rough esteam of merged workload','Min Temp(C)','Min Temp Flag','Cool Deg Days Flag','Total Snow(cm)','Total Snow Flag','Snow on Grnd(cm)','Snow on Grnd Flag','Dir of Max Gust(10s deg)','Dir of Max Gust Flag','Spd of Max Gust(km/h)','Spd of Max Gust Flag','holiday','NATIONAL_HOLIDAY_IND','PROVINCIAL_HOLIDAY_IND','IMPACT_DAY_FLG','DAY_OF_WEEK','WEEK_OF_YEAR','DAY_OF_MONTH','MONTH_NO','YEAR_NO','HolidayFlag','3days_stack_amount'] 
# key_strings = ['date_numeric']
# Select columns containing key strings
X = data[[col for col in data.columns if any(key == col for key in key_strings)]]
y = data["Planned Absences"]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 8)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
ann = tf.keras.models.Sequential()

# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 62, epochs = 200)
# %%
# print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# %% [markdown]
# Therefore, our ANN model predicts that this customer stays in the bank!
# 
# **Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
# 
# **Important note 2:** Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.

# %% [markdown]
# ### Predicting the Test set results

# %%
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# ### Making the Confusion Matrix

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# %%



