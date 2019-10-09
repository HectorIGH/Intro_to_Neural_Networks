##############################################################
# Author: Héctor Iván García Hernández                       #
# Date: 28/09/2019                                           #
##############################################################

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
#%matplotlib qt

# Sigmoid function
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

# Feed forward
def feed_forward(x, w, b):
    a = []
    for i in range(len(w)):
        if (i == 0):
            a.append(  sigmoid( np.dot(x, w[i].T) + b[i] )  )
        else:
            a.append( sigmoid( np.dot(a[i - 1], w[i].T) + b[i] )    )
    return a

# Network error
def net_error(tar, out):
    err = 0.5 * sum( np.power( np.array(tar) - np.array(out), 2 ) )
    return err

# Back propagation
def BP(x, tar, w, b, a):
    l_error = np.empty_like(a)
    n_w = np.empty_like(w)
    n_b = np.empty_like(b)
    for i in range(len(w)):
        if (i == 0):
            l_error[-1] = -(tar - a[-1]) * a[-1] * (1 - a[-1])
        else:
            l_error[-i - 1] = np.dot(l_error[-i] , w[-i]) * a[-i-1] * (1 - a[-i-1])

    for i in range(len(w)):
        if (i == 0):
            x = np.reshape(x, (1, len(x)))
            error = np.reshape(l_error[0], (len(l_error[0]), 1))
            n_w[i] = w[i] - alpha * np.multiply(error , x)
            n_b[i] = b[i] - alpha * l_error[i]
        else:
            n_w[i] = w[i] - alpha * np.reshape(l_error[i], (len(l_error[i]), 1)) * a[i-1]
            n_b[i] = b[i] - alpha * l_error[i]
    return n_w, n_b

# Testing patterns
def testing_patterns(X, w, b):
    print('       MLP result      ')
    print('Pat:          t:      out:')
    for index, x in enumerate(X):
        a = feed_forward(x, w, b)
        clas = a[-1]
        #print('{}. {} ---- {} ----> {:.3f}'.format(index, x, t[index], float(clas)))
        print(index, x, t[index], clas.round(decimals = 4))

# Accuracy
def acc(X, t, w, b):
    accu = 0
    for index, x in enumerate(X):
        t_i = np.array(t[index])
        clas = np.array(feed_forward(x, w, b)[-1].round(decimals = 0))
        if (np.argwhere(t_i > 0).tolist() == np.argwhere(clas > 0).tolist()):
            accu += 1
    print("Accuracy of the model: ", accu/len(X))
    return accu / len(X)

# Graph error
def graph_error(err_vector):
    plt.figure(0)
    plt.plot(err_vector)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('BP algorithm')
    plt.show()

# Decision boundaries
def dec_boundaries(X, t, w, b):
    # Creating mesh
    h = 0.01
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.c_[xx.ravel(), yy.ravel()]
    out = np.zeros(np.shape(Z)[0])

    #Out model
    for i in range(len(out)):
        out[i] = feed_forward(Z[i], w, b)[-1]
        

    out = out.reshape(xx.shape)
    levels = np.linspace(0, 1)
    plt.figure(1)
    plt.contourf(xx, yy, out, levels)
    plt.colorbar()

    # Plotting data
    lis = np.unique(t)
    for i in range(len(t)):
        if (i == 0):
            pos = np.where(t == 0)[0]
            plt.plot(X[pos][:, 0], X[pos][:, 1], 'o', color = 'white', markersize = 15)
        else:
            pos = np.where(t == 1)[0]
            plt.plot(X[pos][:, 0], X[pos][:, 1], 'x', color = 'red', markersize = 15)

    plt.title('Decision boundary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Training patterns and targets
data = pd.read_csv('iris_v.csv').values
#random.shuffle(data)
X = data[:, :4]#len(data[0]) - 1]
t = data[:, 4:]#len(data[0]) - 1]
#print(X)
#print(t)

random.seed(10)
alpha = 0.02
epochs = 1000

in_neurons = 4 # number of input neurons
L_neurons = 3 # number of output neurons
neurons = [6] # array containing the number of neurons for each hidden layer
neurons_in_layers = [in_neurons] + neurons + [L_neurons]

weights = []
bias = []
mu, sigma = 0, 0.1 # mean and standard deviation for weights
for i in range(len(neurons_in_layers) - 1):
    weights.append(np.array([[random.gauss(mu, sigma) for k in range(neurons_in_layers[i])] for j in range(neurons_in_layers[i + 1])]))

for i in range(len(neurons_in_layers)- 1):
    bias.append(np.array([random.gauss(mu, sigma) for k in range(neurons_in_layers[i + 1])]))

err_vector = []

for epoch in range(epochs):
    err = 0
    for index, x in enumerate(X):
        # Feed Forward
        a = feed_forward(x, weights, bias)
        # Network error
        err += net_error(t[index], a[-1])
        # Back propagation
        weights, bias = BP(x, t[index], weights, bias, a)
    err_vector.append(err / X.shape[0])

#print(weights)
#print(bias)

# Testing patterns
testing_patterns(X, weights, bias)

# Getting accuracy
acc(X, t, weights, bias)

# Graph error
graph_error(err_vector)

# Decision boundaries
#dec_boundaries(X, t, weights, bias)
