import csv
import numpy as np
import matplotlib.pyplot as pyplot

def initialize_weigths(layer_sizes, optimizer = 0.01):
    weights = dict()
    for idx, layer_size in enumerate(layer_sizes):
        if idx == 0: continue
        prev_layer_size = layer_sizes[idx - 1]
        # Initializing random weights
        weights['w' + str(idx)] = np.random.randn(layer_size, prev_layer_size) * optimizer
        # Initializing zeros on bias neurons
        weights['b' + str(idx)] = np.zeros((layer_size, 1))
    return weights

    # l1 = 4 x m = w1 * X
    # w1 = 4 x 2
    # X = 2 x m
    # TODO: assert X, weights that are 2-D arrays
def forward_propagate(X, weights, activation_functions, layer_sizes):
    # Assert equality between # of Input Neurons and # of w1 columns
    assert np.size(X, axis=0) == np.size(weights['w1'], axis=1)
    # Assert equality between # of Input Neurons and Input layer size
    assert np.size(X, axis=0) == layer_sizes[0]

    cache = {'a0': X}
    for idx, layer_size in enumerate(layer_sizes):
        # Skip first layer, it's already initialized
        if idx == 0: continue

        ### Assertion block ###
        # Assert equality between # of neurons in [idx] layer and w[idx]
        assert np.shape(weights['w' + str(idx)]) == (layer_size, layer_sizes[idx-1]), \
        "Shape of w%d weigths: %s\nExpected shape: %s" % (idx, np.shape(weights['w' + str(idx)]), \
        (layer_size, layer_sizes[idx-1]))
        # Assert equality between # of neurons in [idx] layer and b[idx]
        assert np.shape(weights['b' + str(idx)]) == (layer_size, 1), \
        "Shape of b%d weigths: %s\nExpected shape: %s" % (idx, np.shape(weights['b' + str(idx)]), \
        (layer_size, 1))

        # Pull activation values from previous layer
        previous_layer_values = cache['a' + str(idx - 1)]
        # Calculate and put in cache activation values from [idx] layer
        layer_values = np.dot(weights['w' + str(idx)], previous_layer_values) + weights['b' + str(idx)]
        cache['z' + str(idx)] = layer_values
        cache['a' + str(idx)] = activate(layer_values, activation_functions[idx - 1])
    output = cache['a' + str(len(layer_sizes) - 1)]
    return output, cache

def linear_backward_step(dA, W, z, a_prev, activation_func):
    m = dA.shape[1]
    dZ = dA * derivative_of_activation(z, activation_func)
    dW = 1/m * np.dot(dZ, a_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims= True)
    dA_prev = np.dot(W.T, dZ)
    return dZ, dW, db, dA_prev

def backward_propagation(output, Y, weights, cache, activation_functions, layer_sizes):
    cost_function = activation_functions[-1]
    cost, cost_derivative = compute_cost(output, Y, cost_function)
    cache['dA' + str(len(layer_sizes) - 1)] = cost_derivative

    for i in reversed(range(1, layer_sizes - 1)):
        dA = cache['dA' + str(i)]
        W = weights['W' + str(i)]
        z = cache['z' + str(i)]
        a_prev = cache['a' + str(i - 1)]
        activation_function = activation_functions[i]
        dZ_temp, dW_temp, db_temp, dA_prev_temp = linear_backward_step(dA, W, z, a_prev, activation_function)
        cache['dZ' + str(i)] = dZ_temp
        cache['dW' + str(i)] = dW_temp
        cache['db' + str(i)] = db_temp
        cache['dA' + str(i - 1)] = dA_prev_temp

def update_parameters(weigths, cache, learning_rate):
    L = len(weights) // 2
    for l in range(1, L + 1):
        weights['W' + str(l)] = weights['W' + str(l)] - learning_rate * cache['dW' + str(l)]
        weights['b' + str(l)] = weights['b' + str(l)] - learning_rate * cache['db' + str(l)]
    return weights

def derivative_of_activation(value, activation_func):
    activation_value = activate(value, activation_func)
    # Sigmoid function
    if activation_func == 'sigmoid':
        return activation_value * (1 - activation_value)
    
    # Tanh function
    elif activation_func == 'tanh':
        return 1 - np.power(activation_value, 2)
    
    # ReLU function
    elif activation_func == 'relu':   
        return np.greater(value, 0).astype(int)
    
    # Leaky ReLU function
    elif activation_func == 'leaky_relu':
        return np.greater(value, 0).astype(int) * 0.01

    # Not implemented
    else:
        raise ValueError('Not implemented activation function')

def compute_cost(output, Y, cost_function):
    m = Y.shape[0]
    
    ### Sum of squared error function ###
    if cost_function == 'sum_of_squared_error':
        pass
    #    cost = 1 / m * np.sum(np.power(Y - output, 1), axis = 1, keepdims = True)
    #    derivative_of_cost = 
    elif cost_function == 'cross_entropy_error':
        cost = -1 / m * np.sum(Y * np.log(output) + (1 - Y) * np.log(1 - output), axis = 1, keepdims=True)
        derivative_of_cost = np.divide(output - Y, output * (1 - output))
    
    cost = np.squeeze(cost)  
    assert(cost.shape == ())
    
    return cost, derivative_of_cost

def activate(value, activation_func='relu'):
    # Sigmoid function
    if activation_func == 'sigmoid':
        return 1 / (1 + np.exp(-value))
    
    # Tanh function
    elif activation_func == 'tanh':
        return np.tanh(value)
    
    # ReLU function
    elif activation_func == 'relu':   
        return value * (value > 0)
    
    # Leaky ReLU function
    elif activation_func == 'leaky_relu':
        return np.maximum(0.01 * value, value)

    # Not implemented
    else:
        raise ValueError('Not implemented activation function')


#### RUNNER ####
#print(initialize_weigths([2, 4, 1]))
#print(activate(np.array([-5, -2.5, 0, 1, 2.5, 100, 1000]), 'relu'))
#print(activate(np.array([-5, -2.5, 0, 1, 2.5, 100, 1000]), 'leaky_relu'))

X = np.array([[1]])
weights = {
    "w1": np.array([[1], [2]]),
    "b1": np.array([[1], [1]]),
    "w2": np.array([[-1, 1]]),
    "b2": np.array([[3]]),
}
layer_sizes=[1,2,1]
activation_functions = ['relu'] * 2
print(forward_propagate(X, weights, activation_functions, layer_sizes))
