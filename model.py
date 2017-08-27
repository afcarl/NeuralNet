import numpy as np

class Sequential:
    def __init__(self):
        self.model = []
        self.y = []
        self.lr = 0.01

    def __call__(self, x_, save=True):
        x = np.zeros((1,len(x_)))
        x[0] = x_
        for layer in self.model:
            #self.y.append(x)
            x = layer(x,save)
        return x

    def add(self, layer):
        self.model.append(layer)

    def compile(self, loss="MSE", optimizer="SGD"):
        self.loss = loss
        self.optimizer = optimizer
        if optimizer == "MomentumSGD":
            self.lr = 0.001
            self.grad = [0. for i in range(len(self.model))]
        elif optimizer == "RMSPropGraves":
            self.eta = 0.00025
            self.alpha = 0.95
            self.beta = 0.95
            self.epsilon = 0.0001
            self.h = [0. for i in range(len(self.model))]
            self.g = [0. for i in range(len(self.model))]
            self.v = [0. for i in range(len(self.model))]
        elif optimizer == "Adam":
            self.lr == 0.001
            self.m = [0. for i in range(len(self.model))]
            self.v = [0. for i in range(len(self.model))]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

    def update(self, loss):
        if self.optimizer == "SGD":
            self.SGD(loss)
        elif self.optimizer == "MomentumSGD":
            self.MomentumSGD(loss)
        elif self.optimizer == "RMSPropGraves":
            self.RMSPropGraves(loss)
        elif self.optimizer == "Adam":
            self.Adam(loss)

    def SGD(self, loss):
        for layer in reversed(self.model):
            grad = np.dot(layer.x.T, loss)
            loss = np.dot(grad, np.ones((grad.shape[1],1))).T
            loss = np.delete(loss,-1,1)
            layer.W -= self.lr * layer.act_d(grad)

    def MomentumSGD(self, loss):
        for i,layer in enumerate(reversed(self.model)):
            grad = np.dot(layer.x.T, loss)
            loss = np.dot(grad, np.ones((grad.shape[1],1))).T
            loss = np.delete(loss,-1,1)
            self.grad[i] = self.lr * grad + 0.9 * self.grad[i]
            layer.W -= self.grad[i]

    def RMSPropGraves(self, loss):
        for i,layer in enumerate(reversed(self.model)):
            grad = np.dot(layer.x.T, loss)
            loss = np.dot(grad, np.ones((grad.shape[1],1))).T
            loss = np.delete(loss,-1,1)
            self.h[i] = self.alpha * self.h[i] + (1.-self.alpha) * (grad**2)
            self.g[i] = self.alpha * self.g[i] + (1.-self.alpha) * grad
            self.v[i] = (-1*self.eta / np.sqrt(self.h[i] - self.g[i]**2 + self.epsilon)) * grad+ self.beta * self.v[i]
            layer.W += self.v[i]

    def Adam(self, loss):
        for i,layer in enumerate(reversed(self.model)):
            grad = np.dot(layer.x.T, loss)
            loss = np.dot(grad, np.ones((grad.shape[1],1))).T
            loss = np.delete(loss,-1,1)
            self.m[i] = self.beta1 * self.m[i] + (1.-self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.-self.beta2) * (grad**2)
            m_ = self.m[i]/(1.-self.beta1)
            v_ = self.v[i]/(1.-self.beta2)
            layer.W -= self.lr * m_ / (np.sqrt(v_) + self.epsilon)
