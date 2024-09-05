import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train_file_path = "C:/Users/vnadi/OneDrive/Documents/train.csv"
test_file_path = "C:/Users/vnadi/OneDrive/Documents/test.csv"
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

X_train = train_data.drop(columns=['SalePrice'])
y_train = train_data['SalePrice'].values
X_test = test_data.drop(columns=['SalePrice'])
y_test = test_data['SalePrice'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class LinearRegressionGDWithL2:
    def __init__(self, learning_rate=0.01, epochs=1000, l2_penalty=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  
        self.theta = np.zeros(X.shape[1])  

        for epoch in range(self.epochs):
            predictions = X.dot(self.theta)
            errors = predictions - y
            gradient = X.T.dot(errors) / len(y)
            self.theta -= self.learning_rate * (gradient + self.l2_penalty * self.theta)
            loss = np.mean(errors ** 2) + self.l2_penalty * np.sum(self.theta ** 2)
            self.loss_history.append(loss)

        return self.theta

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X] 
        return X.dot(self.theta)

lr_gd_l2 = LinearRegressionGDWithL2(learning_rate=0.01, epochs=1000, l2_penalty=0.1)
lr_gd_l2.fit(X_train_scaled, y_train)

y_pred_gd_l2 = lr_gd_l2.predict(X_test_scaled)

mse_gd_l2 = mean_squared_error(y_test, y_pred_gd_l2)

ridge_sklearn = Ridge(alpha=0.1)
ridge_sklearn.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_sklearn.predict(X_test_scaled)

mse_ridge_sklearn = mean_squared_error(y_test, y_pred_ridge)

plt.figure(figsize=(10, 6))
plt.plot(lr_gd_l2.loss_history, label="Gradient Descent with L2 Regularization Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE + Regularization Term)')
plt.title('Loss vs Epochs (with L2 Regularization)')
plt.legend()

plt.show(block=True)

print(f"Gradient Descent with L2 Regularization Model MSE: {mse_gd_l2}")
print(f"SciKit-Learn Ridge Regression Model MSE: {mse_ridge_sklearn}")

