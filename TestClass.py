from neuralnetworkmodel import NeuralNetworkModel
from neuralnetworktrainer import NeuralNetworkTrainer
from neuralnetworktrainer import NeuralNetworkExecutor
from dataviz.datavizualizator import DataVizualizator

"""
Modeling Neural Network Example
"""

### Initialize neural network model ###
nn_model = NeuralNetworkModel()

### Set input layer ###
nn_model.set_input_layer(2)

### Add hidden layers ###
# params: number of units, activation_func (default: relu)
nn_model.add_hidden_layer(4)
nn_model.add_hidden_layer(4)

### Set output layer ###
# params: number of units, activation_func (default: sigmoid)
nn_model.set_output_layer(3)

"""
Configuring Neural Network Trainer
"""
### Initialize neural network trainer ###
nn_trainer = NeuralNetworkTrainer()

### Set MANDATORY training parameters ###
# params: cost_function, learning_rate, number_of_iter
nn_trainer.set_cost_function('cross_entropy')
nn_trainer.set_learning_rate(1)
nn_trainer.set_num_of_iterations(5000)
#nn_trainer.set_num_of_epochs(35000)

### Set regularization training parameters ###
# params: regularization_param, dropout_percentage
#nn_trainer.set_regularization(reg_param)
#nn_trainer.set_dropout_percentage(layer_num, dropout_percentage)
#nn_trainer.set_dropout_percentages(dropout_percentages_list)

### Set optimization parameters
# params: beta1, beta2
#nn_trainer.set_momentum_optimization(beta1)
#nn_trainer.set_adam_optimization(beta1, beta2, epsilon = 1e-8)

### Set batch/stochastic gradient descent parameters ###
# params: minibatch_size
#nn_trainer.set_minibatch_size(64)

### Set Neural Network Model ###
### MUST BE SET BEFORE IMPORTING TRAINING DATA ###
# params: nn_model
nn_trainer.set_model(nn_model)

### Set training data ###
# params: X, Y, datasets_split_ratio
nn_trainer.import_data('izmesano.csv', delimiter = ' ')
#nn_trainer.import_X('input.csv', delimiter=' ')
#nn_trainer.import_Y('output.csv', delimiter=' ')
nn_trainer.set_training_data_split_ratio(1, 0, 3)

### Plot training data ###
X = nn_trainer.get_X()
Y = nn_trainer.get_Y()
#print("X:\n" + str(X))
#print("Y:\n" + str(Y))
DataVizualizator.plot_data(X, Y)

### Normalize training data ###
nn_trainer.multiclass_clasifier()
nn_trainer.normalize_training_data(normalize_Y = False)

### Train neural network ###
nn_trainer.train()

### Initialize Neural Network Executor ###
nn_executor = NeuralNetworkExecutor()
nn_executor.set_trained_neural_network(nn_trainer)

### Predict output based on input data ###
# params: X
X = nn_trainer.get_X()
Y = nn_trainer.get_Y()
output = nn_executor.predict(X)
print("Output:\n" + str(output))

### Measure prediction accuracy ###
predictions = nn_executor.prediction_accuracy(X, Y)
print("Accuracy: %d%%" % predictions)

### Import & Export trained Neural Network ###
# params: filename
#nn_trainer.import_neural_network(filename)
#trained_nn_model = nn_trainer.export_neural_network(filename='')