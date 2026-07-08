import numpy as np
from activations import Activations
from losses import Loss
from optimizers import SGD, Momentum, Adam, Nesterov, AdaGrad, RMSProp, Muon
from weights import WeightInit


# -------------------------------------------------
# MLP Class
# -------------------------------------------------

class MLP:

    def __init__(self, layer_sizes, activations,
                 loss='cross_entropy', learning_rate=0.01,
                 optimizer='sgd', batch_size=32,
                 weight_init='xavier', regularization=None,
                 lambda_reg=0.01):

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        # initialize weights
        self.weights, self.biases = WeightInit.initialize_weights(layer_sizes, weight_init)

        # loss
        if loss == 'mse':
            self.loss = Loss.mse
            self.loss_deriv = Loss.mse_derivative
        else:
            self.loss = Loss.cross_entropy
            self.loss_deriv = Loss.cross_entropy_derivative

        # optimizer selection
        if optimizer == 'momentum':
            self.optimizer = Momentum()
        elif optimizer == 'adam':
            self.optimizer = Adam()
        elif optimizer == 'nesterov':
            self.optimizer = Nesterov()
        elif optimizer == 'adagrad':
            self.optimizer = AdaGrad()
        elif optimizer == 'rmsprop':
            self.optimizer = RMSProp()
        elif optimizer == 'muon':
            self.optimizer = Muon()
        else:
            self.optimizer = SGD()

    # ---------------------------------------------
    # Forward Pass
    # ---------------------------------------------

    def forward(self, X):

        activations = [X]
        zs = []

        A = X

        for i in range(len(self.weights)):

            Z = A @ self.weights[i] + self.biases[i]
            zs.append(Z)

            act = self.activations[i]

            if act == "sigmoid":
                A = Activations.sigmoid(Z)

            elif act == "tanh":
                A = Activations.tanh(Z)

            elif act == "relu":
                A = Activations.relu(Z)

            elif act == "leaky_relu":
                A = Activations.leaky_relu(Z)

            elif act == "linear":
                A = Z

            activations.append(A)

        return activations, zs

    # ---------------------------------------------
    # Backward Pass
    # ---------------------------------------------

    def backward(self, X, y):

        activations, zs = self.forward(X)

        grads_w = []
        grads_b = []

        delta = self.loss_deriv(y, activations[-1])

        for i in reversed(range(len(self.weights))):

            A_prev = activations[i]

            grads_w.insert(0, A_prev.T @ delta)
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))

            if i > 0:

                Z = zs[i-1]
                act = self.activations[i-1]

                if act == "sigmoid":
                    delta = (delta @ self.weights[i].T) * Activations.sigmoid_derivative(Z)

                elif act == "tanh":
                    delta = (delta @ self.weights[i].T) * Activations.tanh_derivative(Z)

                elif act == "relu":
                    delta = (delta @ self.weights[i].T) * Activations.relu_derivative(Z)

                elif act == "leaky_relu":
                    delta = (delta @ self.weights[i].T) * Activations.leaky_relu_derivative(Z)

        return grads_w, grads_b

    # ---------------------------------------------
    # Training
    # ---------------------------------------------

    def fit(self, X_train, y_train, X_val, y_val, epochs=100):

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        for epoch in range(epochs):

            grads_w, grads_b = self.backward(X_train, y_train)

            # update weights
            params = self.weights + self.biases
            grads = grads_w + grads_b
            self.optimizer.update(params, grads, self.learning_rate)

            # predictions
            train_pred, _ = self.forward(X_train)
            val_pred, _ = self.forward(X_val)

            train_loss = self.loss(y_train, train_pred[-1])
            val_loss = self.loss(y_val, val_pred[-1])
            
            train_acc = self.accuracy(y_train, train_pred[-1])
            val_acc = self.accuracy(y_val, val_pred[-1])

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

        return history
    @staticmethod
    def accuracy(y_true, y_pred, tolerance=10):
        correct = np.abs(y_true - y_pred) < tolerance
        return np.mean(correct)