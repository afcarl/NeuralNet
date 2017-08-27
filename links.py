import numpy as np
from numpy.random import *

class Linear:
    def __init__(self, inp_n, out_n, activation=None, initialization=None):
        #self.W = np.r_[np.random.randn(inp_n,out_n) / np.sqrt(inp_n), np.zeros((1,out_n))]
        if initialization == "HeNormal":
            self.W = np.r_[normal(0, np.sqrt(2./inp_n), (inp_n, out_n)), np.zeros((1,out_n))]
        elif initialization == "zeros":
            self.W = np.zeros((inp_n+1,out_n))
        else:
            self.W = np.r_[normal(0, np.sqrt(1./inp_n), (inp_n, out_n)), np.zeros((1,out_n))]
        self.x = None
        self.activation = activation

    def __call__(self, inp, save):
        inp = np.c_[inp,np.ones((1,1))]
        if save:
            self.x = inp
        y = np.dot(inp, self.W)
        if self.activation == "sigmoid":
            return 1. / (1 + np.exp(-1*y))
        elif self.activation == "tanh":
            return np.tanh(y)
        elif self.activation == "relu":
            return y * (y > 0)
        elif self.activation == "softmax":
            e = np.exp(y - np.max(y))  # prevent overflow
            return e / np.array([np.sum(e, axis=1)]).T
        return y

    def act_d(self, x):
        if self.activation == "sigmoid":
            return x * (1. - x)
        elif self.activation == "tanh":
            return 1. - x * x
        elif self.activation == "relu":
            return 1. * (x > 0)
        elif self.activation == "softmax":
            return x * (1. - x)
        return x
