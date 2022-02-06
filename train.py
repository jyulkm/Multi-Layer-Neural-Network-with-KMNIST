################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
from data import write_to_file, z_score_normalize, load_data, load_config
from neuralnet import *


def train(x_train, t_train, x_val, t_val, config, experiment=None):
    """
    Train your model here using batch stochastic gradient descent and early stopping. Use config to set parameters
    for training like learning rate, momentum, etc.

    Args:
        x_train: The train patterns
        t_train: The train labels
        x_val: The validation set patterns
        t_val: The validation set labels
        config: The configs as specified in config.yaml
        experiment: An optional dict parameter for you to specify which experiment you want to run in train.

    Returns:
        5 things:
            training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
            best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    best_model = None

    x_train = z_score_normalize(x_train)

    model = NeuralNetwork(config=config)

    # return train_acc, val_acc, train_loss, val_loss, best_model
    raise NotImplementedError('Train not implemented')


def test(model, x_test, t_test):
    """
    Does a forward pass on the model and returns loss and accuracy on the test set.

    Args:
        model: The trained model to run a forward pass on.
        x_test: The test patterns.
        y_test: The test labels.

    Returns:
        Loss, Test accuracy
    """
    # return loss, accuracy
    raise NotImplementedError('Test not implemented')


def train_mlp(x_train, t_train, x_val, t_val, x_test, t_test, config):
    """
    This function trains a single multi-layer perceptron and plots its performances.

    NOTE: For this function and any of the experiments, feel free to come up with your own ways of saving data
            (i.e. plots, performances, etc.). A recommendation is to save this function's data and each experiment's
            data into separate folders, but this part is up to you.
    """
    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, t_train, x_val, t_val, config)

    test_loss, test_acc = test(best_model, x_test, t_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)


def activation_experiment(x_train, t_train, x_val, t_val, x_test, t_test, config):
    """
    This function tests all the different activation functions available and then plots their performances.
    """
    raise NotImplementedError('Activation Experiment not implemented')


def topology_experiment(x_train, t_train, x_val, t_val, x_test, t_test, config):
    """
    This function tests performance of various network topologies, i.e. making
    the graph narrower and wider by halving and doubling the number of hidden units.

    Then, we change number of hidden layers to 2 of equal size instead of 1, and keep
    number of parameters roughly equal to the number of parameters of the best performing
    model previously.
    """
    raise NotImplementedError('Topology Experiment not implemented')


def regularization_experiment(x_train, t_train, x_val, t_val, x_test, t_test, config):
    """
    This function tests the neural network with regularization.
    """
    raise NotImplementedError('Regularization Experiment not implemented')

# compares this gradient with the math formula


def get_approx(e, model_B, x_train, t_train, backprop_vals):
    # stores the difference of gradients
    diffs = []

    for layer in ["input_hidden", "hidden_output"]:
        for kind in ["weight", "bias"]:
            print(layer + " " + kind + ":")

            if kind == "weight":  # choose two weight values: w_00, w_10
                for i in range(2):
                    model_plus_e = initialize_weight(
                        model_B, layer, kind, e, i, j=0)
                    model_minus_e = initialize_weight(
                        model_B, layer, kind, -e, i, j=0)

                    _, E0 = model_minus_e(x_train, t_train)
                    _, E1 = model_plus_e(x_train, t_train)

                    backprop_val = backprop_vals["_".join(
                        [layer, kind[0], str(i), str(0)])]
                    approx_val = (E1 - E0) / (2*e)

                    print("backprop val: " + str(backprop_val))
                    print("approx val: " + str(approx_val))

                    diff = np.abs(backprop_val - approx_val)

                    diffs.append(diff)

            if kind == "bias":
                model_plus_e = initialize_weight(
                    model_B, layer, kind, e, 0)
                model_minus_e = initialize_weight(
                    model_B, layer, kind, -e, 0)

                _, E0 = model_minus_e(x_train, t_train)
                _, E1 = model_plus_e(x_train, t_train)

                backprop_val = backprop_vals["_".join([layer, kind[0]])]
                approx_val = (E1 - E0) / (2*e)

                print(backprop_val > 0)
                print("approx val: " + str(approx_val*1000000))

                diff = np.abs(backprop_val - approx_val)

                diffs.append(diff)

    return np.all(np.array(diffs) <= e**2)


def initialize_weight(model_B, layer, kind, e=0, i=None, j=None):
    """
    Creates model_A and sets its weights and biases to match those of model_B:
    model_A.w = model_B.w + e
    """
    model_A = NeuralNetwork(config=model_B.config)

    # saves the layers
    input_hidden_A = model_A.layers[0]
    hidden_output_A = model_A.layers[2]

    input_hidden_B = model_B.layers[0]
    hidden_output_B = model_B.layers[2]

    # reinitializes model weights to match the original model's
    input_hidden_A.w = input_hidden_B.w
    hidden_output_A.w = hidden_output_B.w

    # reinitializes model biases to match the original model's
    input_hidden_A.b = input_hidden_B.b
    hidden_output_A.b = hidden_output_B.b

    # change value
    if kind == "weight":
        if layer == "input_hidden":
            input_hidden_A.w[i][j] += e
        if layer == "hidden_output":
            hidden_output_B.w[i][j] += e
    if kind == "bias":
        if layer == "input_hidden":
            input_hidden_A.b[i] += e
        if layer == "hidden_output":
            hidden_output_B.b[i] += e

    return model_A


def check_gradients(x_train, t_train, config):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """

    # normalizes data
    x_train, _ = data.z_score_normalize(x_train)
    t_train = data.one_hot_encoding(t_train)

    # initializes the model
    model = NeuralNetwork(config=config)
    y = model(x_train, t_train)

    # saves the layers of the model
    # Selecting the input to hidden layer object
    input_hidden = model.layers[0]
    # Selecting the hidden to output layer object
    hidden_output = model.layers[2]

    # saves the initial bias terms of each layer
    input_hidden_b0 = input_hidden.b
    hidden_output_b0 = hidden_output.b

    # saves the initial weight matrices of each layer
    input_hidden_w0 = input_hidden.w
    hidden_output_w0 = hidden_output.w

    # performs backward propagation
    model.backward()

    # saves the initial bias terms of each layer
    input_hidden_b1 = input_hidden.b
    hidden_output_b1 = hidden_output.b

    # saves the initial weight matrices of each layer
    input_hidden_w1 = input_hidden.w
    hidden_output_w1 = hidden_output.w

    # calculates the gradients of the weight matrices
    input_hidden_wg = (input_hidden_w1 - input_hidden_w0) / model.learning_rate
    hidden_output_wg = (hidden_output_w1 -
                        hidden_output_w0) / model.learning_rate

    # calculates the gradients of the bias terms
    input_hidden_bg = (input_hidden_b1[0] -
                       input_hidden_b0[0]) / model.learning_rate

    hidden_output_bg = (hidden_output_b1 -
                        hidden_output_b0) / model.learning_rate

    backprop_vals = {
        "input_hidden_w_0_0": input_hidden_wg[0][0],
        "input_hidden_w_1_0": input_hidden_wg[1][0],
        "hidden_output_w_0_0": hidden_output_wg[0][0],
        "hidden_output_w_1_0": hidden_output_wg[1][0],
        "input_hidden_b": input_hidden_bg[0],
        "hidden_output_b": hidden_output_bg[0]
    }

    if get_approx((10 ** -2), model, x_train, t_train, backprop_vals):
        print("Success: gradients within range")
    else:
        print("Fail: gradients not within range")


config = load_config("./config.yaml")
x_train = load_data()[0]
t_train = load_data()[1]

check_gradients(x_train, t_train, config)
