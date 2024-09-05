import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_file_path = r"C:\Users\vnadi\OneDrive\Documents\train.csv"
test_file_path = r"C:\Users\vnadi\OneDrive\Documents\test.csv"
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

X_train = train_data.drop(columns=['SalePrice'])
y_train = train_data['SalePrice'].values
X_test = test_data.drop(columns=['SalePrice'])
y_test = test_data['SalePrice'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

theta = np.linalg.inv(X_train_scaled.T.dot(X_train_scaled)).dot(X_train_scaled.T).dot(y_train)

y_pred_normal_eq = X_test_scaled.dot(theta)

mse_normal_eq = mean_squared_error(y_test, y_pred_normal_eq)

lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test_scaled)

mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

print(f"Normal Equation Model MSE: {mse_normal_eq}")
print(f"SciKit-Learn Linear Regression Model MSE: {mse_sklearn}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_normal_eq, color='blue', label='Normal Equation Predictions', alpha=0.6)
plt.scatter(y_test, y_pred_sklearn, color='red', label='SciKit-Learn Predictions', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--', label="Perfect Fit Line")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Predicted vs Actual Sale Prices")
plt.legend()
plt.show()

