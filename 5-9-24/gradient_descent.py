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

# Linear Regression using Gradient Descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  
        self.theta = np.zeros(X.shape[1])  

        for epoch in range(self.epochs):
            predictions = X.dot(self.theta)
            errors = predictions - y
            gradient = X.T.dot(errors) / len(y)
            self.theta -= self.learning_rate * gradient

            loss = np.mean(errors ** 2)
            self.loss_history.append(loss)

        return self.theta

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  
        return X.dot(self.theta)

lr_gd = LinearRegressionGD(learning_rate=0.01, epochs=1000)
lr_gd.fit(X_train_scaled, y_train)

y_pred_gd = lr_gd.predict(X_test_scaled)

mse_gd = mean_squared_error(y_test, y_pred_gd)

lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test_scaled)

mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

plt.figure(figsize=(10, 6))
plt.plot(lr_gd.loss_history, label="Gradient Descent Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Loss vs Epochs')
plt.legend()
plt.show()

print(f"Gradient Descent Model MSE: {mse_gd}")
print(f"SciKit-Learn Model MSE: {mse_sklearn}")

