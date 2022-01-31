################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################

# TODO: double check layer class forward function
# TODO: double check layer class backward function

import numpy as np
import math


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
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

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
        if x < 0:
            return 0
        if x > 0:
            return 1
        if x == 0:
            return 0  # following the convention of returning 1 only if x > 0


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

    def backward(self, delta, grad_act):
        """
        Takes the weighted sum of the deltas from the layer above it as input.
        Computes gradient for its weights and the delta to pass to its previous layers.
        Return self.d_x
        """
        self.d_w = self.x.T @ delta

        self.d_x = grad_act(self.a) * (self.w @ delta)

        return self.d_x


class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

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
        raise NotImplementedError(
            "Forward propagation not implemented for NeuralNetwork")

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        raise NotImplementedError(
            "Backward propagation not implemented for NeuralNetwork")

    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        return np.exp(a) / np.sum(np.exp(a))

    def loss(self, logits, targets):
        """
        Compute the categorical cross-entropy loss and return it.
        """
        raise NotImplementedError("Loss not implemented for NeuralNetwork")
