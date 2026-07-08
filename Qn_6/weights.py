import numpy as np

class WeightInit:

    @staticmethod
    def initialize_weights(layer_sizes, method="xavier"):
        weights = []
        biases = []

        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]

            if method == "xavier":
                W = np.random.randn(in_dim, out_dim) * np.sqrt(1/in_dim)

            elif method == "he":
                W = np.random.randn(in_dim, out_dim) * np.sqrt(2/in_dim)

            else:  # random
                W = np.random.randn(in_dim, out_dim) * 0.01

            b = np.zeros((1, out_dim))

            weights.append(W)
            biases.append(b)

        return weights, biases