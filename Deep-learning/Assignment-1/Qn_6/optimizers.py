import numpy as np 
# -------------------------------------------------
# Optimizers
# -------------------------------------------------

class SGD:
    def update(self, params, grads, lr):
        for i in range(len(params)):
            params[i] -= lr * grads[i]

class Nesterov:

    def __init__(self, beta=0.9):
        self.beta = beta
        self.v = None

    def update(self, params, grads, lr):

        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        for i in range(len(params)):

            v_prev = self.v[i]
            self.v[i] = self.beta*self.v[i] - lr*grads[i]

            params[i] += -self.beta*v_prev + (1+self.beta)*self.v[i]


class AdaGrad:

    def __init__(self, eps=1e-8):
        self.eps = eps
        self.G = None

    def update(self, params, grads, lr):

        if self.G is None:
            self.G = [np.zeros_like(p) for p in params]

        for i in range(len(params)):

            self.G[i] += grads[i]**2
            params[i] -= lr * grads[i] / (np.sqrt(self.G[i]) + self.eps)

class RMSProp:

    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.Eg = None

    def update(self, params, grads, lr):

        if self.Eg is None:
            self.Eg = [np.zeros_like(p) for p in params]

        for i in range(len(params)):

            self.Eg[i] = self.beta*self.Eg[i] + (1-self.beta)*(grads[i]**2)

            params[i] -= lr * grads[i] / (np.sqrt(self.Eg[i]) + self.eps)



class Muon:

    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.m = None

    def update(self, params, grads, lr):

        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]

        for i in range(len(params)):

            self.m[i] = self.beta*self.m[i] + (1-self.beta)*grads[i]

            params[i] -= lr * self.m[i] / (np.linalg.norm(self.m[i]) + self.eps)
class Momentum:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.v = None

    def update(self, params, grads, lr):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.v[i] = self.beta*self.v[i] + (1-self.beta)*grads[i]
            params[i] -= lr*self.v[i]


class Adam:
    def __init__(self):
        self.m = None
        self.v = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

    def update(self, params, grads, lr):

        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*grads[i]
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*(grads[i]**2)

            m_hat = self.m[i] / (1-self.beta1**self.t)
            v_hat = self.v[i] / (1-self.beta2**self.t)

            params[i] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

