import numpy as np
from logger import logger
from sklearn.metrics import accuracy_score


def layer_dims(X, Y, hidden_layers, output_act_fn='softmax'):
    """
    Create the list of units per layer. The length of the list determines the number of hidden layers.
    :param X: Input vector from the dataset.
    :param Y: Output vector from the dataset.
    :param hidden_layers: List of tuples with the number of units per layer and the activation function.
    :param output_act_fn: Activation function of the output layer.
    :return: The dimensions of each layer with the activation function per layer.
    """
    if type(X) != np.ndarray or type(Y) != np.ndarray or type(hidden_layers) != list:
        raise TypeError("X and Y must be np.ndarray, and hidden_layers must be a list of integer.")

    layers = [(X.shape[0], 'input')] + hidden_layers + [(Y.shape[0], output_act_fn)]

    return layers


def initialize_parameters(layer_dims):
    """
    Initialize weights to a small random float to break symmetry, initialize bias to zero.
    :param layer_dims: Layer dimensions from the function layer_dims()
    :return: Initialized parameters of the neural network, weights and biases.
    """

    parameters = {}
    for idx in range(len(layer_dims)):
        if idx > 0:
            parameters['W' + str(idx)] = np.random.randn(layer_dims[idx][0], layer_dims[idx-1][0])*0.01
            parameters['b' + str(idx)] = np.zeros((layer_dims[idx][0], 1))
    return parameters


def sigmoid(Z):
    """
    Calculate the sigmoid of vector Z.
    :param Z: Input vector.
    :return: Output vector.
    """
    A = 1.0 / (1.0 + np.exp(-Z))
    return A


def tanh(Z):
    pass


def relu(Z):
    """
    Calculate the ReLU value of vector Z.
    :param Z: Input vector.
    :return: Output vector.
    """
    A = Z * (Z > 0)
    return A


def softmax(Z):
    """
    Calculate the softmax value of a matrix Z.
    :param Z: Input matrix.
    :return: Output matrix.
    """
    den = np.sum(np.exp(Z), axis=0, keepdims=True)

    A = np.true_divide(np.exp(Z), den)

    return A


def forward_layer(A_prev, parameters, layer_number, activation_function='relu'):
    """
    Forward propagation thru one layer with the linear function, with weights and bias, and the activation function.
    :param A_prev: Output from previous layer.
    :param parameters: Dictionary of the weights and biases.
    :param layer_number: The layer number.
    :param activation_function: Name of the activation function.
    :return: Output A, output Z from the linear function.
    """
    W = parameters['W' + str(layer_number)]
    b = parameters['b' + str(layer_number)]

    Z = np.dot(W, A_prev) + b
    A = globals()[activation_function](Z)

    return A, Z


def forward_propagation(X, Y, parameters, layers):
    """
    Forward propagation thru all the layers. Also used to calculate predictions on the test set.
    :param X: Input vector.
    :param Y: Output vector.
    :param parameters: Dictionary of weights and biases.
    :param layers: List of tuples for the number of units and activation function for each layer.
    :return: Predicted output, cache of A and Z values, cost value.
    """
    A = [X]
    cache = [(X, None)]

    idx = 1
    while idx < len(layers):
        A_last, Z_last = forward_layer(A[idx-1], parameters, idx, layers[idx][1])
        A.append(A_last)
        cache.append((A_last, Z_last))
        idx += 1

    cost = compute_cost(Y, A[-1])

    return A[-1], cache, cost


def compute_cost(Y, A_final):
    """
    Compute the cost of the predicted output.
    :param Y: True output vector.
    :param A_final: Predicted output vector.
    :return: Cost.
    """
    m = Y.shape[1]
    n_y = Y.shape[0]

    assert(n_y == A_final.shape[0])

    cost = np.sum(Y*np.log(A_final))/m

    return cost


def linear_backward(dZ, A_prev, W):
    """
    A single linear layer backpropagation.
    :param dZ: Derivative of the activation function.
    :param A_prev: Output of the previous layer.
    :param W: Weights of the layer.
    :return: Derivative of the weight, biases, and the outputs of the previous layer.
    """
    m = dZ.shape[1]
    dW = 1.0/m*np.dot(dZ, A_prev.T)
    db = 1.0/m*np.sum(dZ, axis=1, keepdims=True)
    logger.debug((W.T.shape,dZ.shape))
    dA_prev = np.dot(W.T, dZ)

    return dW, db, dA_prev


def back_propagation(AL, Y, parameters, cache, layers):
    """
    Complete backpropagation of the algorithm.
    :param AL: Predicted output vector.
    :param Y: True output vector.
    :param parameters: Dictionary of weight and biases.
    :param cache: Cache of A and Z from forward propagation.
    :param layers: Tuples for the number of units and activation function name for each layer.
    :return: Gradients for weights, biases, and layer outputs.
    """
    L = len(cache)
    grads = {}

    # First step is dAL
    dAL = AL - Y
    grads['dW'+str(L-1)], grads['db'+str(L-1)], grads['dA'+str(L-2)] = linear_backward(dAL, cache[L-2][0], parameters['W'+str(L-1)])

    assert(AL.shape == Y.shape)
    logger.debug([(key, val.shape) for key, val in grads.items()])
    logger.debug([item[0].shape for item in cache])

    for idx in reversed(range(L-2)):
        if layers[idx][1] == 'relu' or layers[idx][1] == 'input':
            dZ_temp = grads['dA'+str(idx+1)] * np.heaviside(cache[idx+1][1], 0)
        else:
            raise TypeError('Only backward propagation for relu function has been implemented yet.')
        logger.debug(grads.keys())
        grads['dW' + str(idx+1)], grads['db' + str(idx+1)], grads['dA' + str(idx)] = linear_backward(dZ_temp, cache[idx][0],
                                                                                               parameters['W' + str(idx+1)])

    logger.debug([(key, val.shape) for key, val in grads.items()])
    return grads


def update_parameters(parameters, grads, nb_of_layers, learning_rate):
    """
    Update weights and biases for next epoch.
    :param parameters: Dictionary of weights and biases.
    :param grads: Gradients for weights and biases.
    :param nb_of_layers: Number of layers.
    :param learning_rate: Learning rate for gradient descent.
    """
    for i in range(1,nb_of_layers-1):
        parameters['W'+str(i)] = parameters['W'+str(i)] - learning_rate*grads['dW'+str(i)]
        parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]


def accuracy_vector(Y, AL):
    """
    Calculate the accuracy of the prediction.
    :param Y: True output vector.
    :param AL: Predicted output vector.
    :return: Accuracy score.
    """
    return accuracy_score(np.argmax(Y, axis=0), np.argmax(AL, axis=0))


def nn_model(X, Y, hidden_layers, n_iterations=20, learning_rate=0.1, X_test=None, Y_test=None):
    """
    L-layer neural network for multioutput classification, gradient-descent optimized using cross-entropy loss function.
    :param X: Input matrix with each instance as a column (n_x x m).
    :param Y: True output matrix with each instance as a column (n_y x m)
    :param hidden_layers: List of tuples for the number of units and the name of the activation function [(300, 'relu'),(150, 'relu')]
    :param n_iterations: Number of epochs.
    :param learning_rate: Learning rate for gradient descent.
    :param X_test: Input matrix of the test dataset.
    :param Y_test: Output matrix of the test dataset.
    :return: Parameters of the last epoch, (Cost, Training set accuracy, validation set accuracy) for each layer.
    """
    layers = layer_dims(X, Y, hidden_layers)
    parameters = initialize_parameters(layers)
    logger.debug(layers)
    logger.debug([(key, val.shape) for key, val in parameters.items()])
    if type(X_test) != None:
        performance_array = np.array([[0.0, 0.0, 0.0]])
    else:
        performance_array = np.array([[0.0, 0.0]])
    # TODO iterate from here
    for i in range(n_iterations):
        AL, cache, cost = forward_propagation(X, Y, parameters, layers)
        logger.info("Iteration #{}, cost function value: {}".format(i, cost))
        grads = back_propagation(AL, Y, parameters, cache, layers)
        update_parameters(parameters, grads, len(layers), learning_rate)
        logger.debug([(key, val.shape) for key, val in grads.items()])
        logger.debug([Y[:,0], AL[:,0]])
        train_acc = accuracy_vector(Y, AL)
        logger.info("Training set accuracy score: {}".format(train_acc))
        if type(X_test) != None:
            AL_test, cache_test, cost_test = forward_propagation(X_test, Y_test, parameters, layers)
            test_acc = accuracy_vector(Y_test, AL_test)
            logger.info("Test set accuracy score: {}".format(test_acc))
            performance_array = np.append(performance_array, [[cost, train_acc, test_acc]], axis=0)
        else:
            performance_array = np.append(performance_array, [[cost, train_acc]], axis=0)
    return parameters, performance_array
