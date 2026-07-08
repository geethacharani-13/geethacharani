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
        elif loss == "mae":
            self.loss = Loss.mae
            self.loss_deriv = Loss.mae_derivative
        elif loss == "huber":
            self.loss = Loss.huber
            self.loss_deriv = Loss.huber_derivative
        
            

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
        if self.regularization == "l1":
            for j in range(len(grads_w)):
                grads_w[j] += self.lambda_reg * np.sign(self.weights[j])

        elif self.regularization == "l2":
            for j in range(len(grads_w)):
                grads_w[j] += self.lambda_reg * self.weights[j]
        return grads_w, grads_b

    # ---------------------------------------------
    # Training
    # ---------------------------------------------

    def fit(self, X_train, y_train, X_val, y_val, epochs=100):

        history = {
            "train_loss": [],
            "val_loss": [],
            "update_mag": []
        }

        n = X_train.shape[0]
        if self.batch_size is None:
            batch_size = n
        else:
            batch_size = self.batch_size
        for epoch in range(epochs):

            # shuffle every epoch
            indices = np.random.permutation(n)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # mini-batch training
            update_magnitudes=[]
            for i in range(0, n, batch_size):

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                grads_w, grads_b = self.backward(X_batch, y_batch)
                
                params = self.weights + self.biases
                grads = grads_w + grads_b
                old_params = [p.copy() for p in params]

                self.optimizer.update(params, grads, self.learning_rate)

                update_mag = 0
                for old, new in zip(old_params, params):
                     update_mag += np.linalg.norm(new - old)

                update_magnitudes.append(update_mag)
                

            # predictions
            train_pred, _ = self.forward(X_train)
            val_pred, _ = self.forward(X_val)

            train_loss = self.loss(y_train, train_pred[-1])
            val_loss = self.loss(y_val, val_pred[-1])
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["update_mag"].append(np.mean(update_magnitudes))

        return history