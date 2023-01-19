# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:49:06 2019

@author: Yhq
"""

import numpy as np
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [1]])

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1. - sigmoid(x))

def get_weights(shape=()):
    np.random.seed(seed=0)
    return np.random.normal(loc=0.0, scale=0.001, size=shape, )

class Network(object):
    def __init__(self, lr=0.01):
        self.W1 = get_weights((3, 4))
        self.b1 = get_weights((4))
        self.W2 = get_weights((4, 1))
        self.b2 = get_weights((1))
        self.lr = lr

    def backward(self, X_batch, y_batch):
        batch_size = 0
        batch_loss = 0
        batch_acc  = 0

        self.grads_W2 = np.zeros_like(self.W2)
        self.grads_b2 = np.zeros_like(self.b2)
        self.grads_W1 = np.zeros_like(self.W1)
        self.grads_b1 = np.zeros_like(self.b1)

        for x, y in zip(X_batch, y_batch):
            ## forward
            z1  = np.matmul(x, self.W1) + self.b1 # z^{L-1}
            a1  = sigmoid(z1) # a^{L-1}
            z2  = np.matmul(a1, self.W2) + self.b2 # z^{L}
            a2 = sigmoid(z2) # a^{L}

            ## loss
            # 1 : mse
            # loss = 1/2 * np.mean((y-a2)**2) # d_C/ d_a^{L} = (a^{L} - y)
            # delta_L = (a2 - y) * (a2 * (1-a2))
            # 2 : abs
            # loss = np.mean(abs(y - a2))  # |x|' = (1 if x>0 else -1)
            # delta_L = (1. if (a2 - y > 0) else -1.) * a2 * (1-a2)
            # 3 : sigmoid cross entropy (logistic regression)
            loss = - (y*np.log(a2) + (1-y)*np.log(1-a2)) 
            delta_L = - (y * (1-a2) - (1-y) * a2 )
            #print(delta_L)
            
            delta_l = np.matmul(self.W2, delta_L) * a1 * (1-a1)
            #print(delta_l)
        
            self.grads_W2 += np.array([(a1 * delta_L)]).T
            #print(self.grads_W2)
            self.grads_b2 += delta_L
            #print(self.grads_b2)
            self.grads_W1 += np.matmul(np.array([x]).T, [delta_l])
            #print(self.grads_W1)
            self.grads_b1 += delta_l
            #print(self.grads_b1)

            batch_size += 1
            batch_loss += loss
            batch_acc += 1 if (y == (1 if a2>0.5 else 0)) else 0

        self.grads_W2 /= batch_size
        self.grads_b2 /= batch_size
        self.grads_W1 /= batch_size
        self.grads_b1 /= batch_size
        batch_loss /= batch_size
        batch_acc /= batch_size
        print("loss:{} batch_acc:{} batch_size:{} lr:{}".format(batch_loss, batch_acc, batch_size, self.lr))

        self.W2 -= self.lr * self.grads_W2
        self.b2 -= self.lr * self.grads_b2
        self.W1 -= self.lr * self.grads_W1
        self.b1 -= self.lr * self.grads_b1

net = Network(lr=10)

for i in range(100):
    net.backward(X, y)
