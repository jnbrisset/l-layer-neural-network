import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml
from logger import logger
from llayernn import *
from sklearn.utils import Bunch


np.random.seed(42)

def map_outputs(y_vec, unique_values):
    """
    Create a label binarizer for each possible outputs, in this case with an output value of 0 to unique_values-1. In the MNIST, possible values are integer from 0 to 9.
    :param y_vec: One-dimensional vector of y
    :param unique_values: Number of unique values
    :return: Matrix with dimensions unique values (n_y) by number of instances (m)
    """
    y_matrix = np.zeros((len(y_vec), unique_values), np.uint8)

    for idx, val in enumerate(y_vec):
        y_matrix[idx, val] = 1.0

    return y_matrix.T


# Load the data if saved, or fetch the data.
if os.path.exists('data/mnist_data.pkl') and os.path.exists('data/mnist_target.pkl'):
    mnist = {'data': pd.read_pickle('data/mnist_data.pkl')}
    mnist['target'] = pd.read_pickle('data/mnist_target.pkl')
else:
    mnist = fetch_openml('mnist_784', version=1, as_frame=True)
    mnist['data'].to_pickle('data/mnist_data.pkl')
    mnist['target'].to_pickle('data/mnist_target.pkl')

logger.info('Dataset loaded...')

# Load the data to a numpy array, and split the training and test data.
X, y = mnist["data"].to_numpy().astype(np.single), mnist["target"].to_numpy().astype(np.uint8)
X_train, X_test, Y_train, Y_test = X[:60000].T, X[60000:].T, map_outputs(y[:60000], 10), map_outputs(y[60000:], 10)

assert(X_train.shape == (784, 60000))
assert(Y_train.shape == (10, 60000))

# Launch the neural network algorithm.
parameters, performance_data = nn_model(X_train, Y_train, [(50, 'relu'), (30, 'relu'), [20, 'relu']],
                                        n_iterations=100, learning_rate=0.2, X_test=X_test, Y_test=Y_test, backprop_check=5)

# Save the weights and biases, and the performance output.
np.save("data/parameters", parameters)
np.save("data/performance", performance_data)
