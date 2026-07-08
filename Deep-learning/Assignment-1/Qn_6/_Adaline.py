import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Adaline:
    
    def __init__(self, learning_rate=1.0, max_iterations=1000):
        """
        Initialize ADALINE
        learning_rate: step size for weight updates
        max_iterations: maximum training iterations
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.w = None
        self.b = None


    def fit(self, X, y,X_val=None,y_val = None):
        """
        Train ADALINE using batch gradient descent
        X: (n_samples, n_features)
        y: (n_samples,)
        Returns: training history (MSE per epoch) and validation history (MSE per epoch)
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
     
        train_history = []
        val_history = []
     
        for _ in range(self.max_iterations):
          y_pred = np.dot(X, self.w) + self.b
          errors = y - y_pred
        
          self.w += self.learning_rate * np.dot(X.T, errors) / n_samples
          self.b += self.learning_rate * np.mean(errors)
        
          train_mse = np.mean(errors**2)
          if np.isnan(train_mse) or np.isinf(train_mse) or train_mse > 1e10:
            print("Diverged.")
            break
          train_history.append(train_mse)
        
          if X_val is not None:
            val_pred = np.dot(X_val, self.w) + self.b
            val_mse = np.mean((y_val - val_pred)**2)
            val_history.append(val_mse)
    
        return train_history, val_history


    def predict(self, X):
        """Return predicted outputs (linear output)"""
        return np.dot(X, self.w) + self.b


    def score(self, X, y):
        """Return mean squared error"""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
    