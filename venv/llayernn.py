import numpy as np
from logger import logger
from sklearn.metrics import accuracy_score


class Opt_params:
    def __init__(self, layers, optimization = 'adam'):
        self.nb_of_layers = len(layers)-1

        opt_types = {"none", 'adam', 'rms-prop', 'momentum'}

        self.V, self.S = {}, {}

        if optimization.lower() not in opt_types or type(optimization) is not str:
            logger.warn('Optimization algorithm "{}" is not supported. Adam optimization used.')
            self.optimization = 'adam'
        else:
            self.optimization = optimization.lower()

        for idx in range(self.nb_of_layers + 1):
            if idx > 0:
                if self.optimization in ['adam', 'momentum']:
                    self.V['dW' + str(idx)] = np.zeros((layers[idx][0], layers[idx - 1][0]))
                    self.V['db' + str(idx)] = np.zeros((layers[idx][0], 1))
                else:
                    self.V['dW' + str(idx)] = np.ones((layers[idx][0], layers[idx - 1][0]))
                    self.V['db' + str(idx)] = np.ones((layers[idx][0], 1))

                if self.optimization in ['adam', 'rms-prop']:
                    self.S['dW' + str(idx)] = np.zeros((layers[idx][0], layers[idx - 1][0]))
                    self.S['db' + str(idx)] = np.zeros((layers[idx][0], 1))
                else:
                    self.S['dW' + str(idx)] = np.ones((layers[idx][0], layers[idx - 1][0]))
                    self.S['db' + str(idx)] = np.ones((layers[idx][0], 1))

    def grad_update(self, grad_name, grad, iteration, beta1=0.9, beta2=0.999, epsilon=1e-8):
        assert(grad.shape == self.V[grad_name].shape and grad.shape == self.S[grad_name].shape)

        if self.optimization == 'none':
            return grad

        if self.optimization == 'momentum':
            self.V[grad_name] = beta1 * self.V[grad_name] + (1 - beta1) * grad
            return self.V[grad_name] / (1-(beta1 ** iteration))

        if self.optimization == 'rms-prop':
            self.S[grad_name] = beta2 * self.S[grad_name] + (1 - beta2) * np.square(grad)
            return np.divide(grad, np.sqrt(self.S[grad_name]+epsilon))

        if self.optimization == 'adam':
            self.V[grad_name] = beta1*self.V[grad_name] + (1-beta1)*grad
            self.S[grad_name] = beta2 * self.S[grad_name] + (1 - beta2) * np.square(grad)
            return np.divide(self.V[grad_name], np.sqrt(self.S[grad_name])+epsilon)


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


def initialize_parameters(layer_dims, optimization):
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
    return parameters, Opt_params(layer_dims, optimization)


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

    cost = -1.0*np.sum(Y*np.log(A_final))/m

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

    for idx in reversed(range(L-2)):
        if layers[idx][1] == 'relu' or layers[idx][1] == 'input':
            dZ_temp = grads['dA'+str(idx+1)] * np.heaviside(cache[idx+1][1], 0)
        else:
            raise TypeError('Only backward propagation for relu function has been implemented yet.')
        grads['dW' + str(idx+1)], grads['db' + str(idx+1)], grads['dA' + str(idx)] = linear_backward(dZ_temp, cache[idx][0],
                                                                                               parameters['W' + str(idx+1)])
    grads.pop('dA0')

    return grads


def update_parameters(parameters, grads, opt_params, iteration, nb_of_layers, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update weights and biases for next epoch.
    :param parameters: Dictionary of weights and biases.
    :param grads: Gradients for weights and biases.
    :param nb_of_layers: Number of layers.
    :param learning_rate: Learning rate for gradient descent.
    """

    for i in range(1,nb_of_layers-1):
        dW_update = opt_params.grad_update('dW'+str(i),
                                               grads['dW'+str(i)],
                                               iteration+1,
                                               beta1,
                                               beta2,
                                               epsilon)
        db_update = opt_params.grad_update('db' + str(i),
                                           grads['db' + str(i)],
                                           iteration + 1,
                                           beta1,
                                           beta2,
                                           epsilon)
        parameters['W'+str(i)] = parameters['W'+str(i)] - learning_rate * dW_update
        parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * db_update


def accuracy_vector(Y, AL):
    """
    Calculate the accuracy of the prediction.
    :param Y: True output vector.
    :param AL: Predicted output vector.
    :return: Accuracy score.
    """
    return accuracy_score(np.argmax(Y, axis=0), np.argmax(AL, axis=0))


def params_to_vector(parameters, nb_of_layers):
    vector = []
    for idx in range(1, nb_of_layers+1):
        vector = np.append(vector, parameters['W' + str(idx)].reshape(-1))
        vector = np.append(vector, parameters['b' + str(idx)].reshape(-1))
    return vector


def grads_to_vector(grads, nb_of_layers):
    vector = []
    for idx in range(1, nb_of_layers+1):
        vector = np.append(vector, grads['dW' + str(idx)].reshape(-1))
        vector = np.append(vector, grads['db' + str(idx)].reshape(-1))
    return vector


def vector_to_params(params_vector, layers):
    parameters = {}
    idx_counter = 0
    for idx in range(1, len(layers)):
        parameters['W' + str(idx)] = params_vector[idx_counter:idx_counter+np.multiply(layers[idx][0], layers[idx-1][0])].reshape(layers[idx][0], layers[idx-1][0])
        idx_counter += np.multiply(layers[idx][0], layers[idx-1][0])
        parameters['b' + str(idx)] = params_vector[idx_counter:idx_counter+layers[idx][0]].reshape(layers[idx][0], 1)
        idx_counter += layers[idx][0]
    return parameters


def vector_to_grads(grads_vector, layers):
    grads = {}
    idx_counter = 0
    for idx in range(1, len(layers)):
        grads['dW' + str(idx)] = grads_vector[idx_counter:idx_counter+np.multiply(layers[idx][0], layers[idx-1][0])].reshape(layers[idx][0], layers[idx-1][0])
        idx_counter += np.multiply(layers[idx][0], layers[idx-1][0])
        grads['db' + str(idx)] = grads_vector[idx_counter:idx_counter+layers[idx][0]].reshape(layers[idx][0], 1)
        idx_counter += layers[idx][0]
    return grads


def gradient_check(parameters, X, Y, layers, epsilon = 1e-7, nb_of_samples = 5):
    random_idx = np.random.randint(0, X.shape[1], nb_of_samples)
    X_sample, Y_sample = np.array([X[:, random_idx[0]]]).T, np.array([Y[:, random_idx[0]]]).T
    for idx in range(1, nb_of_samples):
        X_sample = np.concatenate((X_sample, np.array([X[:, idx]]).T), axis=1)
        Y_sample = np.concatenate((Y_sample, np.array([Y[:, idx]]).T), axis=1)

    AL, cache, _ = forward_propagation(X_sample, Y_sample, parameters, layers)
    grads = back_propagation(AL, Y_sample, parameters, cache, layers)

    params_vector = params_to_vector(parameters, len(layers)-1)
    grads_vector = grads_to_vector(grads, len(layers)-1)
    num_parameters = params_vector.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters, 1))
    grads_approx = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        thetaplus = np.copy(params_vector)
        thetaplus[i] = thetaplus[i] + epsilon
        _, _, J_plus[i] = forward_propagation(X_sample, Y_sample, vector_to_params(thetaplus, layers), layers)

        thetaminus = np.copy(params_vector)
        thetaminus[i] = thetaminus[i] - epsilon
        _, _, J_minus[i] = forward_propagation(X_sample, Y_sample, vector_to_params(thetaminus, layers), layers)

        grads_approx[i] = (J_plus[i] - J_minus[i])/2/epsilon

        if i % 10000 == 0 and i != 0:
            logger.info("{}th variable out of {}.".format(i, num_parameters))

    num = np.linalg.norm(grads_vector.reshape(-1) - grads_approx.reshape(-1))
    den = np.linalg.norm(grads_vector.reshape(-1)) + np.linalg.norm(grads_approx.reshape(-1))
    difference = num/den

    if difference > 2e-7 or difference == np.nan:
        print("There is a mistake in the backward propagation! difference = " + str(difference))
        logger.debug([(idx, grads_vector[idx], grads_approx[idx], theta) for idx, theta in
                      enumerate((grads_vector.reshape(-1) - grads_approx.reshape(-1)) / den) if theta > 1e-2])
    else:
        print("Your backward propagation works perfectly fine! difference = " + str(difference))

    return difference


def minibatch_initialization(X, Y, minibatch_sz):
    if minibatch_sz == None:
        return np.array([X]), np.array([Y])

    permuted_idx = np.random.permutation(list(range(X.shape[1])))
    X_permuted = np.array([X[:,idx] for idx in permuted_idx]).T
    Y_permuted = np.array([Y[:, idx] for idx in permuted_idx]).T

    assert(X.shape == X_permuted.shape and Y.shape == Y_permuted.shape)

    nb_of_minibatch = X.shape[1] // minibatch_sz
    X_minibatch, Y_minibatch = [], []

    for idx in range(nb_of_minibatch):
        X_minibatch.append(X_permuted[:, idx*minibatch_sz:(idx+1)*minibatch_sz])
        Y_minibatch.append(Y_permuted[:, idx * minibatch_sz:(idx + 1) * minibatch_sz])

    X_minibatch.append(X_permuted[:, -X.shape[1]%minibatch_sz:])
    Y_minibatch.append(Y_permuted[:, -X.shape[1] % minibatch_sz:])

    #assert(X_minibatch.shape == (nb_of_minibatch+1, X.shape[0], minibatch_sz))

    #logger.debug(X_minibatch.shape)

    return X_minibatch, Y_minibatch


def nn_model(X_train, Y_train, hidden_layers, n_iterations=20, learning_rate=0.1, X_test=None, Y_test=None, backprop_check=-1,
             minibatch_sz = None, optimization='adam', beta1=0.9, beta2=0.999, epsilon=1e-8):
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
    layers = layer_dims(X_train, Y_train, hidden_layers)
    parameters, opt_params = initialize_parameters(layers, optimization)
    logger.debug(layers)
    logger.debug([(key, val.shape) for key, val in parameters.items()])
    if type(X_test) != None:
        performance_array = np.array([[0.0, 0.0, 0.0]])
    else:
        performance_array = np.array([[0.0, 0.0]])

    for i in range(n_iterations):
        X, Y = minibatch_initialization(X_train, Y_train, minibatch_sz)
        print('Epoch #{}'.format(i+1), end='  ')
        percent_count = 1
        nb_of_minibatch = len(X)
        for idx, minibatch in enumerate(X):
            AL, cache, cost = forward_propagation(X[idx], Y[idx], parameters, layers)
            grads = back_propagation(AL, Y[idx], parameters, cache, layers)
            update_parameters(parameters, grads, opt_params, i*len(X) + idx, len(layers), learning_rate, beta1, beta2, epsilon)
            # logger.debug([idx, cost])
            if idx // (percent_count*0.1*nb_of_minibatch) > 0:
                print("{}%".format(percent_count*10), end='  ' if percent_count < 9 else '\n')
                percent_count += 1
        AL, _, cost = forward_propagation(X_train, Y_train, parameters, layers)
        train_acc = accuracy_vector(Y_train, AL)
        logger.info("Epoch #{}, cost function value: {}".format(i+1, cost))
        logger.info("Training set accuracy score: {}".format(train_acc))
        if type(X_test) != None:
            AL_test, _, _ = forward_propagation(X_test, Y_test, parameters, layers)
            test_acc = accuracy_vector(Y_test, AL_test)
            logger.info("Test set accuracy score: {}".format(test_acc))
            performance_array = np.append(performance_array, [[cost, train_acc, test_acc]], axis=0)
        else:
            performance_array = np.append(performance_array, [[cost, train_acc]], axis=0)
        if i==backprop_check:
            logger.info("Checking gradient... May take a while...")
            gradient_check(parameters, X_train, Y_train, layers, epsilon=1e-5)
    return parameters, performance_array
