import numpy as np

''' Dropout '''
class Dropout:
    def __init__(self, dropout_percent):
        self.dropout_percent = dropout_percent
    
    def __call__(self, x):
        self.mask = np.random.binomial(1, 1-self.dropout_percent, size=x.shape)[0]
        self.mask = self.mask / (1 - self.dropout_percent)
        x *= self.mask
        return x
    
    def backward(self, dtop):
        dtop *= self.mask
        return dtop

''' Activation Functions '''
class Relu:
    ''' Output shape == Input shape '''
    def __call__(self, x):
        self.x = x
        self.x [self.x < 0] = 0
        return self.x
    
    def backward(self, dtop):
        dtop [self.x <= 0] = 0
        return dtop   

class Softmax:
    ''' Output shape == Input shape '''
    def __call__(self, x, dim):
        self.x = x
        x = np.clip(self.x, -700, 700)
        x = np.exp(x)
        x = x / np.sum(x, axis=dim, keepdims=True)
        self.pred = x
        return self.pred
    
    def backward(self, y):
        dx = self.pred - y
        return dx
    
def sigmoid(x):
    x = np.clip(x, -700, 700)
    x = 1 / (1 + np.exp(-x))
    return x

def sigmoid_backward(x):
    x = x * (1 - x)
    return x

def tanh(x):
    x = np.clip(x, -350, 350)
    x = 2 / (1 + np.exp(-2 * x)) - 1
    return x

def tanh_backward(x):
    x = 1 - np.square(x)
    return x  

''' Loss Function'''
def cross_entropy(pred, y):
    loss = y * np.log(pred + 1e-6)
    loss = -np.sum(loss, axis=1, keepdims=True)
    loss = np.mean(loss)
    return loss