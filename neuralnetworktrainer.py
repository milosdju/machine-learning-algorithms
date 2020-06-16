import math
import numpy as np
import pandas
import matplotlib.pyplot as plt

"""
TODO: ADD VALIDATOR CLASS!
"""
class NeuralNetworkTrainer():
    """ 
    Initialization method for Neural Network Trainer
    """
    def __init__(self):
        # Neural Network model
        self.model = None

        # Mandatory training parameters (Hyperparameters)
        self.__cost_function = None
        self.__learning_rate = None
        self.__num_of_iterations = None
        self.__num_of_epochs = None

        # Optimization training parameters
        self.__minibatch_size = None

        # Train / dev / test datasets
        self.__training_data_split_ratio = None
        self.__train_dataset = None
        self.__dev_dataset = None
        self.__test_dateset = None

        self.__training_data = None
        self.__expected_data = None
        self.__num_of_training_samples = None

        # Training/Cost history
        self.__cost_history = list()
        
    def set_model(self, neural_network_model):
        # TODO: PROVERA tipa!!!
        self.model = neural_network_model

    """ 
    Forward and backward propagation methods
    """
    def __forward_propagation_step(X, W, b, activation_func):
        Z = np.dot(W, X) + b
        A, daZ = NeuralNetworkTrainer.__activate(Z, activation_func)
        return Z, A, daZ

    def __forward_propagation(self, input_data = None):
        L = len(self.model.get_layer_sizes())

        # Set first layer activation values
        if input_data is None:
            input_data = self.__train_dataset
        self.model.state.set_activation_values(0, input_data)

        # Propagate through other layers
        for l in range(1, L):
            X = self.model.state.get_activation_values(l - 1)
            W, b = self.model.state.get_weights(l)
            activation_func = self.model.get_activation_func(l - 1)

            # Compute layer and activation values
            Z_temp, A_temp, daZ_temp = NeuralNetworkTrainer.__forward_propagation_step(X, W, b, activation_func)
            self.model.state.set_layer_values(l, Z_temp)
            self.model.state.set_activation_values(l, A_temp) 
            self.model.state.set_activation_derivative_values(l, daZ_temp)

        # Return calculated output
        self.model.state.set_calculated_output(A_temp)
        return A_temp

    def __backward_propagation_step(dA, daZ, W, Z, A_prev):
        # Export number of training samples
        m = dA.shape[1]
        dZ = dA * daZ
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis = 1, keepdims= True)
        dA_prev = np.dot(W.T, dZ)
        return dZ, dW, db, dA_prev

    def __backward_propagation(self, output_data = None, Y = None):
        if output_data is None:
            output_data = self.model.state.get_calculated_output()
        
        if Y is None:
            Y = self.__expected_data

        # Number of layers
        L = len(self.model.get_layer_sizes())
        cost_function = self.__cost_function
        cost, derivative_of_cost = NeuralNetworkTrainer.__compute_cost(output_data, Y, cost_function)
        self.model.state.set_activation_gradient_values(L - 1, derivative_of_cost)

        for l in reversed(range(1, L)):
            dA = self.model.state.get_activation_gradient_values(l)
            daZ = self.model.state.get_activation_derivative_values(l)
            W, b = self.model.state.get_weights(l)
            Z = self.model.state.get_layer_values(l)
            A_prev = self.model.state.get_activation_values(l - 1)

            # Compute derivative of layer, activation and weights values
            dZ_temp, dW_temp, db_temp, dA_prev_temp = \
            NeuralNetworkTrainer.__backward_propagation_step(dA, daZ, W, Z, A_prev)
            self.model.state.set_layer_gradient_values(l, dZ_temp)
            self.model.state.set_weight_gradient_values(l, dW_temp, db_temp)
            self.model.state.set_activation_gradient_values(l - 1, dA_prev_temp)
        
        # Return starting cost
        return cost

    def execute(self, input_data):
        return self.__forward_propagation(input_data)

    """ 
    Training methods
    """
    def set_X(self, X):
        # TODO: Split input data into 3 categories:
        # traning data, cross-validation data and test data
        # according to inserted ratios
        
        # Number of input parameters should be equal to
        # number of non-bias units in input layer 
        assert X.shape[1] == self.model.get_layer_sizes()[0], \
        "Input data size (%d) doesn't match input layer size (%d)" % (X.shape[1], self.model.get_layer_sizes[0])
        
        # Set number of training samples
        self.__num_of_training_samples = X.shape[0]

        # Training data will be in the following format:
        # (num_of_units x num_of_training_samples)
        self.__training_data = X.T

    def get_X(self):
        return self.__training_data.T

    def set_Y(self, Y):
        # TODO: WTF???
        # Expected data should be 1-D vector
        #assert Y.ndim == 1, "Y should be 1-D vector"

        # Correct 1D/2D dimensionality
        Y = np.reshape(Y, (Y.shape[0], -1))

        # Expected data will be in the following format:
        # (1 x num_of_training_samples)
        self.__expected_data = Y.T

    def get_Y(self):
        return self.__expected_data.T

    # TODO: multiclass Y
    def import_data(self, filename, delimiter = ',', header = None, include_labels = True):
        training_data = pandas.read_csv(filename, header = header, delimiter = delimiter).as_matrix()
        if include_labels:
            self.set_X(training_data[:, 0:-1])
            self.set_Y(training_data[:, -1])
        else:
            self.set_X(training_data)

    def import_X(self, filename, delimiter = ',', header = None):
        X = pandas.read_csv(filename, header = header, delimiter = delimiter).as_matrix()
        self.set_X(X)

    def import_Y(self, filename, delimiter = ',', header = None):
        Y = pandas.read_csv(filename, header = header, delimiter = delimiter).as_matrix()

        # Squeeze Y to ?-D array
        Y = np.squeeze(Y)
        self.set_Y(Y)

    # Now, it just perform mean normalization 
    # TODO:  add range to rescale
    def normalize_training_data(self, normalize_X = True, normalize_Y = True):
        if self.__training_data is None:
            raise Exception("[PRE] Training data is not set")
        if self.__expected_data is None:
            raise Exception("[PRE] Expected data is not set")

        # Normalize training data
        if normalize_X:
            training_data = self.__training_data
            training_data = (training_data - np.mean(training_data)) / (np.max(training_data) - np.min(training_data))
            self.__training_data = training_data

        if normalize_Y:
            expected_data = self.__expected_data
            expected_data = (expected_data - np.mean(expected_data)) / (np.max(expected_data) - np.min(expected_data))
            self.__expected_data = expected_data

    def multiclass_clasifier(self):
        if self.__expected_data is None:
            raise Exception("[PRE] Expected data is not set")

        multiclass_Y = self.convert_to_bit_classes(self.__expected_data)
        self.__expected_data = multiclass_Y.T
        print("[PRE] Expected data is transformed according to multiclass rules")

    def convert_to_bit_classes(self, Y):
        bit_Y = np.equal(Y.T, np.unique(Y)).astype(int)
        return bit_Y

    def __default_training_data_split_ratio(self):
        raise Exception("Not implemented yet")

    def set_training_data_split_ratio(self, train, dev, test):
        self.__training_data_split_ratio = (train, dev, test)

    def __split_data_into_datasets(self):
        ### TODO: OPTIMIZE THIS - USE ONLY WHAT's NEEDED - REST LEAVE ON FILE SYSTEM ###
        train, dev, test = self.__training_data_split_ratio
        print("[PRE] Spliting training data into train/dev/test sets with following ratio:\
        %d-%d-%d" % (train, dev, test))

        total = train + dev + test
        print("TOTAL " + str(total))
        print("TRAIN " + str(train))
        trainPartSize = round(train / total * self.__num_of_training_samples)
        print("TRAIN PART SIZE PRE " + str(trainPartSize))
        devPartSize = round(dev / total * self.__num_of_training_samples)
        testPartSize = round(test / total * self.__num_of_training_samples)
        
        # Following case: 1-1-1 ratio brings 0.33+0.33+0.33 < 1
        if trainPartSize + devPartSize + testPartSize < self.__num_of_training_samples:
            trainPartSize = self.__num_of_training_samples - (devPartSize + testPartSize)
        

        print("TRAIN PART SIZE POST " + str(trainPartSize))
        ### TODO: This probably going to be removed due to above TODO ###
        print("[PRE] Loading train dataset...")
        self.__train_dataset = (self.__training_data[:, :trainPartSize], self.__expected_data[:, :trainPartSize])
        self.__dev_dataset = (self.__training_data[:, trainPartSize:(trainPartSize + devPartSize)], self.__expected_data[:, trainPartSize:(trainPartSize + devPartSize)])
        self.__test_dateset = (self.__training_data[:, (trainPartSize + devPartSize):], self.__expected_data[:, (trainPartSize + devPartSize):]) 
        print("[PRE] Train dataset size: %d" % (trainPartSize))
        print("[PRE] Dev dataset size: %d" % (devPartSize))
        print("[PRE] Test dataset size: %d" % (testPartSize))
        print("[PRE] Spliting training data into datasets finished")

    """ 
    Training methods
    """
    def set_cost_function(self, cost_function):
        self.__cost_function = cost_function

    def set_learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

    def set_num_of_iterations(self, num_of_iterations):
        self.__num_of_iterations = num_of_iterations

    def set_num_of_epochs(self, num_of_epochs):
        self.__num_of_epochs = num_of_epochs

    def get_minibatch_size(self):
        if self.__minibatch_size is None:
            return self.__train_dataset[0].shape[1]
            #return self.__num_of_training_samples
        return self.__minibatch_size
    
    def set_minibatch_size(self, minibatch_size):
        self.__minibatch_size = minibatch_size

    def __compute_cost(output, Y, cost_function):
        #print("output is " + str(output))
        #print("output shape is " + str(output.shape))
        #print("Y is " + str(Y))
        #print("Y shape is " + str(Y.shape))
        m = np.size(Y, axis = 1)
        cost = 0
        derivative_of_cost = 0

        ### Sum of squared error function ###
        if cost_function == 'sum_of_squared':
            raise Exception("%s is not implemented yet" % cost_function)
        #    cost = 1 / m * np.sum(np.power(Y - output, 2), axis = 1, keepdims = True)
        #    derivative_of_cost =
        
        ### Cross-entropy error function ### 
        elif cost_function == 'cross_entropy':
            cost = -1 / m * np.sum(Y * np.log(output) + (1 - Y) * np.log(1 - output), axis = 1, keepdims=True)
            derivative_of_cost = np.divide(output - Y, output * (1 - output))
        
        cost = np.squeeze(cost)  
        return cost, derivative_of_cost

    def __save_cost_history(self, cost, num_of_iteration, print_at = 100):
        avg_cost = np.mean(cost)
        self.__cost_history.append(avg_cost)
        
        # Determine iteration type
        if self.__num_of_iterations is not None:
            iteration_type = 'iteration'
        else:
            iteration_type = 'epoch'
        
        if num_of_iteration % print_at == 0:
            print("[TRAIN] Cost at %s %d is %f" % (iteration_type, num_of_iteration, avg_cost))

    def __draw_cost_function_line(self):
        plt.plot(self.__cost_history)
        plt.xlabel('# of iterations')
        plt.ylabel('Cost')
        plt.ylim(bottom = 0)
        plt.show()

    def training_checklist(self): 
        print("[PRE] Pre-training checklist started")
        ### Neural Network model validity checking ###
        #assert self.model.check_validity(), "[PRE] Neural Network model is not set properly"

        ### Training parameters checking ###
        assert self.__train_dataset is not None, "[PRE] Train dataset should be initialized"
        assert self.__expected_data is not None, "[PRE] Expected data should be initialized"
        assert self.__cost_function is not None, "[PRE] Cost function should be initialized"
        assert self.__learning_rate is not None, "[PRE] Learning rate should be initialized"
        assert self.__num_of_iterations is not None or self.__num_of_epochs is not None, \
        "[PRE] One of # of iterations or # of epochs should be initialized"

        ### Training data checking ###
        # Number of expected data should be equal to
        # number of training samples
        assert np.size(self.__expected_data, axis = 1) == self.__num_of_training_samples, \
        "[PRE] Number of expected samples (%d) doesn't match number of training samples (%d)" \
        % (np.size(self.__expected_data, axis = 1), self.__num_of_training_samples)

        # Number of expected data dimensionality should be equal to
        # number of output layer units
        assert np.size(self.__expected_data, axis = 0) == self.model.get_layer_sizes()[-1], \
        "[PRE] Number of expected data dimensionality (%d) doesn't match number of output units (%d)" \
        % (np.size(self.__expected_data, axis = 0), self.model.get_layer_sizes()[-1])

        ### Training optimizers checking ###
        assert self.__num_of_iterations is None or self.__num_of_epochs is None, \
        "[PRE] Just one of # of iterations or # of epochs should be initialized"

        ### Checklist finished ###
        print("[PRE] Pre-training checklist finished")

        ### Logger statements ###
        print("[PRE-LOG] Layer sizes: ", self.model.get_layer_sizes())
        print("[PRE-LOG] Activation functions: ", self.model.get_activation_functions())
        print("[PRE-LOG] Number of training samples: ", self.__num_of_training_samples)
        print("[PRE-LOG] Cost function: %s" % self.__cost_function)
        print("[PRE-LOG] Learning rate (alpha): %f" % self.__learning_rate)
        
        if self.__num_of_iterations is not None:
            print("[PRE-LOG] Number of training iterations: %d" % self.__num_of_iterations)
        elif self.__num_of_epochs is not None:
            print("[PRE-LOG] Number of training epochs: %d" % self.__num_of_epochs)

    def __update_parameters(self):
        L = len(self.model.get_layer_sizes())
        for l in range(1, L):
            W, b = self.model.state.get_weights(l)
            dW, db = self.model.state.get_weight_gradient_values(l)
            learning_rate = self.__learning_rate
            self.model.state.set_weights(l, W - learning_rate * dW, b - learning_rate * db)

    def __set_up_minibatches(self, dataset, minibatch_size):
        minibatches = list()

        # Shuffle training data samples
        #shuffled_indices = np.arange(self.__num_of_training_samples)
        dataset_size = dataset[0].shape[1]
        shuffled_indices = np.arange(dataset_size)
        #np.random.shuffle(shuffled_indices)

        # Group training samples into minibatch groups
        for minibatch_idx in range(dataset_size // minibatch_size):
            minibatch_shuffled_indices = \
            shuffled_indices[minibatch_idx * minibatch_size : (minibatch_idx + 1) * minibatch_size]
            minibatches.append((dataset[0][:, minibatch_shuffled_indices], \
            dataset[1][:, minibatch_shuffled_indices]))

        # Deal with remained training samples that couldn't form complete minibatch
        remainings = self.__num_of_training_samples % minibatch_size
        if remainings != 0:
            #print("remainings: " + str(remainings))
            #print("training_data: " + str(self.__training_data))
            #print("expected_data: " + str(self.__expected_data))
            #print("m-input: " + str(self.__training_data[:, minibatch_shuffled_indices[-remainings:]]))
            #print("m-output: " + str(self.__expected_data[:, minibatch_shuffled_indices[-remainings:]]))
            minibatches.append((self.__train_dataset[0][:, minibatch_shuffled_indices[-remainings:]], \
            self.__expected_data[:, minibatch_shuffled_indices[-remainings:]]))

        return minibatches

    # 1 iteration = training model over 1 minibatch of training dataset
    def __iteration_training(self, num_of_iterations):
        minibatch_size = self.get_minibatch_size()
        num_of_minibatches = math.ceil(self.__train_dataset[0].shape[1] / minibatch_size)

        print('num of mini ' + str(num_of_minibatches))

        for i in range(self.__num_of_iterations):
            # Shuffle minibatches
            # EXPLANATION: after each epoch, in order to reduce overfitting, minibatches are shuffled
            # to not allow model to train on order to training data
            if (i % num_of_minibatches == 0):
                train_minibatches = self.__set_up_minibatches(self.__train_dataset, minibatch_size)

            # Perform forward and backward propagation
            self.__forward_propagation(input_data = train_minibatches[i % num_of_minibatches][0])
            cost = self.__backward_propagation(Y = train_minibatches[i % num_of_minibatches][1])

            # Calculate validation set cost
            cost, _ = NeuralNetworkTrainer.__compute_cost(output_data, validation_minibatches[i % num_of_minibatches][1], self.__cost_function)

            # Update weight parameters
            self.__update_parameters()

            # Store cost per iteration in cost_history
            self.__save_cost_history(cost, i)

    # 1 epoch = training model over whole training dataset
    # 1 epoch = size of whole training dataset = num of iterations x batch size 
    def __epochs_training(self, num_of_epochs):
        for i in range(num_of_epochs):
            # Shuffle minibatches
            # EXPLANATION: after each epoch, in order to reduce overfitting, minibatches are shuffled
            # to not allow model to train on order to training data
            minibatch_size = self.get_minibatch_size()
            minibatches = self.__set_up_minibatches(self.__train_dataset, minibatch_size)
            for minibatch in minibatches:
                #print("minibatch: " + str(minibatch))
                # Perform forward and backward propagation
                self.__forward_propagation(minibatch[0])
                cost = self.__backward_propagation(Y = minibatch[1])

                # Update weight parameters
                self.__update_parameters()

                # Store cost per epoch in cost_history
                self.__save_cost_history(cost, i)


    def train(self):
        # Split training data into train/dev/test datasets
        self.__split_data_into_datasets()

        # Pre-training checklist
        self.training_checklist()
        print("[TRAIN] Model training started")

        # Determine iteration type and run training function
        if self.__num_of_iterations is not None:
            self.__iteration_training(self.__num_of_iterations)
        else:
            self.__epochs_training(self.__num_of_epochs)

        print("[TRAIN] Model training finished")
        self.__draw_cost_function_line()
    
    """ 
    Helper methods
    """
    def __activate(value, activation_func):
        ### Sigmoid function ###
        if activation_func == 'sigmoid':
            activation_value = 1 / (1 + np.exp(-value))
            derivative = activation_value * (1 - activation_value)
        
        ### Tanh function ###
        elif activation_func == 'tanh':
            activation_value = np.tanh(value)
            derivative = 1 - np.power(activation_value, 2)
        
        ### ReLU function ###
        elif activation_func == 'relu':   
            activation_value = value * (value > 0)
            derivative = np.greater(value, 0).astype(int)
        
        ### Leaky ReLU function ###
        elif activation_func == 'leaky_relu':
            activation_value = np.maximum(0.01 * value, value)
            derivative = np.greater(value, 0).astype(int) * 0.01

        ### Not implemented ###
        else:
            raise ValueError('Not implemented activation function')

        return activation_value, derivative

class NeuralNetworkExecutor():
    """ 
    Initialization method for Neural Network Executor
    """
    def __init__(self):
        self.trained_neural_network = None

        # List of output activation functions that need adjustment
        self.__activation_functions_for_adjustments = [
            'sigmoid', 'soft_max'
        ]
    
    def set_trained_neural_network(self, trained_neural_network):
        if not isinstance(trained_neural_network, NeuralNetworkTrainer):
            raise Exception("[POST] Inserted trained neural network is invalid")
        self.trained_neural_network = trained_neural_network
    
    """
    Prediction methods
    """
    def predict(self, X):   
        #assert self.__model_initialization_finished, "[POST] Model initialization is not finished"
        output = self.trained_neural_network.execute(X.T)

        # Adjust output results if condition is met
        output_activation_func = self.trained_neural_network.model.get_activation_functions()[-1]
        if output_activation_func in self.__activation_functions_for_adjustments:
            output = self.__adjust_output_result(output, output_activation_func)

        return output
    
    """
    @params: X, Y as #_of_training_samples x #_of_features
    """
    def prediction_accuracy(self, X, Y):
        output = self.predict(X)

        # Adjust output results if condition is met
        output_activation_func = self.trained_neural_network.model.get_activation_functions()[-1]
        if output_activation_func in self.__activation_functions_for_adjustments:
            output = self.__adjust_output_result(output, output_activation_func)

        return np.mean(np.equal(Y.T, output)) * 100

    def __adjust_output_result(self, output, output_activation_func):
        ### Sigmoid ###
        if output_activation_func == 'sigmoid':
            return np.greater_equal(output, 0.5).astype('int')
        ### Soft Max ###
        elif output_activation_func == 'softmax':
            return np.argmax(output, axis = 1)
        ### Not implemented ###
        else:
            raise ValueError('Not implemented adjusment function for output layer')