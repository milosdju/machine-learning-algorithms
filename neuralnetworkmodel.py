import numpy as np
import pandas
import matplotlib.pyplot as plt

class NeuralNetworkModel():
    """ 
    Initialization method for Neural Network
    """
    def __init__(self):
        # Model parameters
        self.__layer_sizes = list()
        self.__activation_functions = list()

        # State parameters
        self.state = NeuralNetworkState()

        # Default optimize parameter for initializing weights
        self.__default_optimizer = 1

        # After model initialization finishes no one layer can be added
        self.__model_initialization_finished = False

    """
    Method for modeling Neural Network
    """
    def set_input_layer(self, num_of_units):
        if self.__layer_sizes != []:
            raise Exception("Input layer is already set")
        self.__layer_sizes.append(num_of_units)

    def add_hidden_layer(self, num_of_units, activation_func = 'sigmoid'):
        if self.__layer_sizes == []:
            raise Exception("Input layer need to be set first")
        elif self.__model_initialization_finished:
            raise Exception("Output layer is set after which no one layer can be added")
        self.__layer_sizes.append(num_of_units)
        self.__activation_functions.append(activation_func)
        W, b = self.__initialize_weigths(self.__layer_sizes[-1], self.__layer_sizes[-2], self.__default_optimizer)
        self.state.set_weights(len(self.__layer_sizes) - 1, W, b)

    def set_output_layer(self, num_of_units, activation_func = 'sigmoid'):
        self.add_hidden_layer(num_of_units, activation_func)
        self.__model_initialization_finished = True

    """ 
    Helper methods
    """
    def __initialize_weigths(self, num_of_units, num_of_prev_units, optimizer):
        # Initializing random weights
        W = np.random.rand(num_of_units, num_of_prev_units) * optimizer
        # Initializing zeros on bias neurons
        b = np.zeros((num_of_units, 1))
        return W, b

    # layer sizes
    def get_layer_sizes(self):
        return self.__layer_sizes

    # Rest    
    def get_activation_functions(self):
        return self.__activation_functions

    def get_activation_func(self, idx):
        return self.__activation_functions[idx]


class NeuralNetworkState:
    
    def __init__(self):
        # Neural Network State parameters
        self.__weights = dict()
        self.__layer_values = dict()
        self.__activation_values = dict()
        self.__calculated_output = None

        # Training state parameters
        self.__num_of_training_samples = 0
        self.__training_data = None
        self.__expected_data = None
        self.__weight_grads = dict()
        self.__layer_grads = dict()
        self.__activation_grads = dict()

        # Training log parameters
        self.__cost_history = list()

    # Z
    def get_layer_values(self, idx):
        return self.__layer_values["Z" + str(idx)]

    def set_layer_values(self, idx, values):
        self.__layer_values["Z" + str(idx)] = values

    # dZ
    def get_layer_gradient_values(self, idx):
        return self.__layer_grads["dZ" + str(idx)]

    def set_layer_gradient_values(self, idx, values):
        self.__layer_grads["dZ" + str(idx)] = values

    # A
    def get_activation_values(self, idx):
        return self.__activation_values["A" + str(idx)]

    def set_activation_values(self, idx, values):
        self.__activation_values["A" + str(idx)] = values

    # daZ
    def get_activation_derivative_values(self, idx):
        return self.__activation_values["daZ" + str(idx)]

    def set_activation_derivative_values(self, idx, values):
        self.__activation_values["daZ" + str(idx)] = values

    # dA
    def get_activation_gradient_values(self, idx):
        return self.__activation_grads["dA" + str(idx)]

    def set_activation_gradient_values(self, idx, values):
        self.__activation_grads["dA" + str(idx)] = values

    # W, b
    def get_weights(self, idx):
        return self.__weights["W" + str(idx)], self.__weights["b" + str(idx)]

    def set_weights(self, idx, W_values, b_values):
        self.__weights["W" + str(idx)] = W_values
        self.__weights["b" + str(idx)] = b_values

    # dW, db
    def get_weight_gradient_values(self, idx):
        return self.__weight_grads["dW" + str(idx)], self.__weight_grads["db" + str(idx)]

    def set_weight_gradient_values(self, idx, dW_values, db_values):
        self.__weight_grads["dW" + str(idx)] = dW_values
        self.__weight_grads["db" + str(idx)] = db_values

    # calculated output
    def get_calculated_output(self):
        return self.__calculated_output

    def set_calculated_output(self, calculated_output):
        self.__calculated_output = calculated_output

    # cost history
    def get_cost_history(self):
        return self.__cost_history