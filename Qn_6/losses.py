import numpy as np

class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return (2/len(y_true)) * (y_pred - y_true)
    
    @staticmethod
    def mae(y, y_hat):
        return np.mean(np.abs(y - y_hat))
    @staticmethod
    def mae_derivative(y, y_hat):
        return np.sign(y_hat - y)


    # HUBER LOSS
    @staticmethod
    def huber(y, y_hat, delta=1.0):
        error = y - y_hat
        mask = np.abs(error) <= delta
        return np.mean(np.where(mask, 0.5 * error**2,
                            delta * (np.abs(error) - 0.5 * delta)))
    @staticmethod
    def huber_derivative(y, y_hat, delta=1.0):
        error = y_hat - y
        mask = np.abs(error) <= delta
        return np.where(mask, error, delta * np.sign(error))