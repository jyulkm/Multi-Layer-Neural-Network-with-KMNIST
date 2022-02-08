################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################

import numpy as np
import math
import data
import copy


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(
                "%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.a = None
        # Placeholder for input dimension.
        self.n = 0

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        self.a = a
        self.d = len(a)
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Takes in the weighted sum delta. 
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return np.multiply(grad, delta)

    def sigmoid(self, a):
        """
        Implement the sigmoid activation here.
        """
        return 1 / (1 + np.exp(-1 * (a)))

    def tanh(self, a):
        """
        Implement tanh here.
        """
        return np.tanh(a)

    def ReLU(self, a):
        """
        Implement ReLU here.
        Takes the max of a_i and 0 for each element i.
        """
        return (a > 0) * a

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.a) * (1 - self.sigmoid(self.a))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - (self.tanh(self.a))**2

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return (self.a > 0).astype(int)


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        # Save the output of forward pass in this (without activation)
        self.a = None

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

        self.prev_w = 0 # For updating weights using momentum
        self.prev_b = 0

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = self.x @ self.w + self.b

        return self.a

    def backward(self, delta, reg=False, reg_type=None, reg_penalty=None):
        """
        Takes an array of deltas from the layer above it as input.
        Computes gradient for its weights and the delta to pass to its previous layers.
        Return self.d_x
        """
        if reg:
            if reg_type == 'L2':
                self.d_w = self.x.T @ delta + 2 * reg_penalty * self.w

            if reg_type == 'L1':
                self.d_w = self.x.T @ delta + reg_penalty * np.sign(self.w)

        else:
            self.d_w = self.x.T @ delta

        self.d_x = delta @ self.w.T

        self.d_b = np.sum(delta.T, axis=1)

        return self.d_x

    def update_weights(self, alpha, gamma):
        momentum_w = gamma * self.d_w + ((1 - gamma) * self.prev_w)
        momentum_b = gamma * self.d_b + ((1 - gamma) * self.prev_b)
        self.prev_w = momentum_w
        self.prev_b = momentum_b
        self.w = self.w + alpha * momentum_w
        self.b = self.b + alpha * momentum_b


class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config, reg=False, reg_type=None):
        """
        Create the Neural Network using config.
        """
        assert (reg_type == 'L2') | (reg_type == 'L1') | (reg_type == None), "Regularization should be L1 or L2"
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.loss = None

        self.config = config
        self.learning_rate = self.config['learning_rate']
        self.gamma = self.config['momentum_gamma'] # momentum term

        self.reg = reg # whether to perform regularization
        self.reg_type = reg_type # type of regularization to perform
        self.reg_penalty = None
        if reg:
            self.reg_penalty = self.config[reg_type + '_penalty']

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(
                Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """

        self.x = x
        self.targets = targets

        z = self.x
        for i in range(0, len(self.layers), 2):
            # example of self.layers: [layer1, activation, layer2]
            a = self.layers[i](z)
            if (i + 1) < len(self.layers):
                z = self.layers[i+1](a)

        a_protect = a - np.max(a)
        self.y = self.softmax(a_protect)

        if self.targets is not None:
            # implement using the cross entropy function below
            self.loss = self.targets - self.y
            return self.y, self.cross_entropy(self.y, self.targets)

        return self.y

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        i = len(self.layers) - 1
        delta_k = self.targets - self.y

        return self.backward_recur(i, delta_k)

    def backward_recur(self, i, delta_k):
        '''
        Delta is the right shape:
            - N x c the first pass
            - N x M the second pass
        '''
        self.layers[i].backward(delta_k, self.reg, self.reg_type, self.reg_penalty)

        self.layers[i].update_weights(self.learning_rate, self.gamma)

        # activation backward
        if i > 0:
            weighted_sum_delta = self.layers[i].d_x
            delta_j = self.layers[i-1].backward(weighted_sum_delta)

            return self.backward_recur(i-2, delta_j)
        
        return self.layers[i].d_w

    def softmax(self, a):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        #assert a.shape==tuple([48000, 10]), "Input matrix must be of shape (48000, 10)"
        return (np.exp(a).T / np.sum(np.exp(a),axis=1)).T

    def cross_entropy(self, logits, targets):
        """
        Compute the categorical cross-entropy loss and return it.
        """
        if self.reg:
            reg_sum = 0
            for i in range(0, len(self.layers), 2):
                if self.reg_type == 'L1':
                    reg_sum += np.sum(np.abs(self.layers[i].w))
                if self.reg_type == 'L2':
                    reg_sum += np.sum(np.square(self.layers[i].w))


            return -1 * np.sum(targets * np.log(logits)) + self.reg_penalty * reg_sum

        else:
            return -1 * np.sum(targets * np.log(logits))

