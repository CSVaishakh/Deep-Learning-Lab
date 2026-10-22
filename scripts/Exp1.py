import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv("Salary.csv")

X = df.loc[:, 'Salary'].values
y = df.loc[:, 'YearsExperience'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)


model = LinearRegression().fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_true=y_test, y_pred=y_pred)

print("Mean Squared Error : ", round(mse, 3))


plt.scatter(X, y, label="Original Data", alpha=0.5)
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression Line")

plt.title("Linear Regression")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.show()



