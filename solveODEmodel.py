import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('TKAgg')
matplotlib.rcParams.update({'font.size': 15})

import matplotlib.pyplot as plt

import time

np.random.seed(1234)
tf.set_random_seed(1234)

class NeuralNetwork:
    def __init__(self, X, y, y0, layers, lb, ub):

        self.lb = lb    # lower bound
        self.ub = ub    # upper bound

        self.X = X
        self.y = y
        self.y0 = y0

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_neural_network(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.X_tf = tf.placeholder(tf.float32, shape=[None, self.X.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.y0_tf = tf.placeholder(tf.float32, shape=[None, self.y0.shape[1]])

        self.y_pred = self.network_y(self.X_tf, self.y0_tf)
        self.dy_pred = self.network_dy(self.X_tf, self.y0_tf)

        self.loss = 1 * tf.reduce_mean(tf.square(self.y_tf - self.y_pred)) + \
                    0 * tf.reduce_mean(tf.square(self.dy_pred - self.f(self.X_tf,self.y_tf)))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_neural_network(self, layers):
        '''
        Initialize weights and biases for each hidden layer
        :param layers: number of hidden layers
        :return: weights and biases
        '''
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_network(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0  # normalize input over [-1, 1]
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def network_y(self, x, y0):
        N = self.neural_network(x, self.weights, self.biases)
        y = y0 + x * N   # A + xN(x)
        return y

    def network_dy(self, x, y0):
        N = self.network_y(x, y0)
        dN = tf.gradients(N, x)[0]
        dy = N + x * dN   # N(x) + x * dN/dx
        return dy

    def f(self, x, y):
        a = x + (1 + 3*x**2)/(1 + x + x**3)
        b = x**3 + 2*x + x**2*(1 + 3*x**2)/(1 + x + x**3)
        # a = 1 / 5
        # b = tf.multiply(tf.exp(tf.div(-x, 5)), tf.cos(x))
        return -a*y + b

    def callback(self, loss):
        print('Loss', loss)

    def train(self):
        tf_dict = {self.X_tf: self.X, self.y_tf: self.y, self.y0_tf: self.y0}

        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

    def predict(self, Xstar, y0star):
        ystar = self.sess.run(self.y_pred, {self.X_tf: Xstar, self.y0_tf: y0star})
        return ystar

if __name__ == "__main__":
    num_hidden_layers = 1
    num_nodes_per_layer = 10

    Psia = lambda x: np.exp(x**2/2)/(1 + x + x**3) + x**2
    # Psia = lambda x: np.exp(-x / 5) * np.sin(x)
    noise = 0
    Ntrain = 10
    N = 100
    lb, ub = 0, 1

    layers = [1]
    for _ in range(num_hidden_layers):
        layers.append(num_nodes_per_layer)
    layers.append(1)

    Xtrain = np.linspace(lb, ub, Ntrain).reshape((Ntrain, 1))
    ytrain = np.array([Psia(x) + noise * np.random.normal(0, 0.1) for x in Xtrain]).reshape((Ntrain, 1))
    y0train = Psia(0) * np.ones((Ntrain, 1), dtype='float32')

    X = np.linspace(lb, 2*ub, N).reshape((N, 1))
    y = np.array([Psia(x) for x in X]).reshape((N, 1))
    y0 = Psia(0) * np.ones((N, 1), dtype='float32')

    model = NeuralNetwork(Xtrain, ytrain, y0train, layers, lb, ub)

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %4f' % (elapsed))

    prediction = model.predict(X, y0)

    plt.figure(figsize=(10, 5))
    plt.plot(X, y, color='red', label='Analytical solution')
    plt.plot(X, prediction, color='black', label='Prediction')
    plt.scatter(Xtrain, ytrain, color='blue', label='Training points')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(X, prediction - y)
    plt.plot([lb,2*ub], [0,0])
    plt.ylim([-0.14, 0.02])
    plt.legend()
    plt.tight_layout()
    plt.show()