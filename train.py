################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
from data import write_to_file, z_score_normalize
from neuralnet import *
import copy


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
    # With momentum: v(t)=alpha*v(t-1)-epsilon*dW(t), w = w + v(t)
    
    batch_size = config['batch_size']
    epochs = config['epochs']
    early_stop_bool = config['early_stop']
    gamma = config['momentum_gamma']
    
    patience = 5 # number of epochs to wait till early stopping if validation error continues to go up
    recent_loss = float('inf')

    model = NeuralNetwork(config=config)
    
    count = 0
    for e in range(epochs):

      minibatch_loss = []  # saves the loss over all minibatches
      minibatch_acc = []  # saves the accuracy over all minibatches
      for minibatch in data.generate_minibatches(x_train, t_train, batch_size):
        x, t = minibatch
        # loss, accuracy = test(model, x, t)  # performs forward propagation
        y, deltas = model(x, t)
        
        model.backward()
        
      y_labels, loss = model(x_val, t_val)
      accuracy = np.mean(np.argmax(y_labels, axis=1) == np.argmax(t_val, axis=1))

      minibatch_loss.append(loss)
      minibatch_acc.append(accuracy)
      
      train_los, train_ac = test(model, x_train, t_train)
      train_acc.append(train_ac)
      train_loss.append(train_los)
      
      val_los, val_ac = test(model, x_val, t_val)
      val_acc.append(val_ac)
      val_loss.append(val_los)

      if loss > recent_loss:
        count += 1
        if count == patience:
            break
      else:
        count = 0
        best_model = model
        recent_loss = loss

    # return train_acc, val_acc, train_loss, val_loss, best_model
    return train_acc, val_acc, train_loss, val_loss, model
    
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
    y_labels, loss = model(x_test, t_test)
    #loss = - 1 * np.sum(t_test * np.log(y_labels))
    accuracy = np.mean(np.argmax(y_labels, axis=1) == np.argmax(t_test, axis=1))

    return loss, accuracy

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
    print("Test Loss", test_loss / x_test.shape[0])
    print("Test Accuracy", test_acc)
    print("Train Accuracy", np.mean(train_acc))
    print("Val Accuracy", np.mean(valid_acc))

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


def update_weight(model, layer, kind, e=0, i=0, j=0):
    """
    Creates model_A and sets its weights and biases to match those of model_B:
    model_A.w = model_B.w + e
    """

    layer_map = {
        "input_to_hidden": 0,
        "hidden_to_output": 2
    }

    if kind == "weight":
        model.layers[layer_map[layer]].w[i][j] += e

    if kind == "bias":
        model.layers[layer_map[layer]].b[0][i] += e


def find_grad_diff(config, layer, kind, x_train, t_train, e, i=0, j=0):
    # initializes the model
    model = NeuralNetwork(config=config)

    update_weight(model, layer, kind, e=e, i=i, j=j)
    _, loss1 = model(x_train, t_train)

    update_weight(model, layer, kind, e=-2*e, i=i, j=j)
    _, loss2 = model(x_train, t_train)
    approx_grad = abs((loss1 - loss2) / (2*e))

    update_weight(model, layer, kind, e=e, i=i, j=j)
    model(x_train, t_train)
    model.backward()

    layer_map = {
        "input_to_hidden": 0,
        "hidden_to_output": 2
    }

    if kind == "weight":
        actual_grad = abs(model.layers[layer_map[layer]].d_w[i][j])
    if kind == "bias":
        actual_grad = abs(model.layers[layer_map[layer]].d_b[0])

    print(layer, kind, i, j)
    print("approximated gradient: ", approx_grad)
    print("back propagation gradient: ", actual_grad)
    print(np.isclose(approx_grad, actual_grad, e**2))
    print()


def check_gradients(x_train, t_train, config):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """
    e = 10**-2

    x_train = np.array([x_train[0]])
    t_train = np.array([t_train[0]])

    for layer in ["input_to_hidden", "hidden_to_output"]:
        for kind in ["weight", "bias"]:
            if kind == "weight":
                for i in [0, 1]:
                    find_grad_diff(config, layer, kind, x_train, t_train, e, i)
            if kind == "bias":
                find_grad_diff(config, layer, kind, x_train, t_train, e)
'''
def check_gradients(x_train, t_train, config):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """
    batch_size = 5
    # initializes the model
    model = NeuralNetwork(config=config)
    y, _ = model(x_train[:batch_size], t_train[:batch_size])

    # saves the layers of the model
    # Selecting the input to hidden layer object
    input_hidden = model.layers[0]
    # Selecting the hidden to output layer object
    hidden_output = model.layers[2]

    w_holder = copy.deepcopy(input_hidden.w)
    b_holder = copy.deepcopy(input_hidden.b)

    # performs backward propagation
    dw = model.backward()

    backprop_vals = {
        "input_hidden_w_0_0": input_hidden.d_w[0][0],
        "input_hidden_w_1_0": input_hidden.d_w[1][0],
        "hidden_output_w_0_0": hidden_output.d_w[0][0],
        "hidden_output_w_1_0": hidden_output.d_w[1][0],
        "input_hidden_b": input_hidden.d_b[0],
        "hidden_output_b": hidden_output.d_b[0]
    }
    print(backprop_vals)

    new_model1 = NeuralNetwork(config=config)

    new_model1.layers[0].w = copy.deepcopy(w_holder)
    new_model1.layers[0].b = copy.deepcopy(b_holder)
    new_model1.layers[0].w[1][0] += 10**-2

    new_model2 = NeuralNetwork(config=config)

    new_model2.layers[0].w = copy.deepcopy(w_holder)
    new_model2.layers[0].b = copy.deepcopy(b_holder)
    new_model2.layers[0].w[1][0] -= 10**-2

    new_y1, new_delta_k1 = new_model1(x_train[6:7], t_train[6:7])
    new_y2, new_delta_k2 = new_model2(x_train[6:7], t_train[6:7])
    print('y: ' + str((new_delta_k1+new_y1)))
    temp1 = -(1/new_y1.shape[1]) * np.sum(np.multiply((new_delta_k1+new_y1),np.log(new_y1)) +
        np.multiply(1-(new_delta_k1+new_y1),np.log(1-new_y1)))
    temp2 = -(1/new_y2.shape[1]) * np.sum(np.multiply((new_delta_k2+new_y2),np.log(new_y2)) +
        np.multiply(1-(new_delta_k2+new_y2),np.log(1-new_y2)))
    print((temp1-temp2)/(2*(10**-2)))
    loss1 = (-1/new_y1.shape[1]) * np.sum((new_delta_k1+new_model1.y)*np.log(new_model1.y))
    loss2 = (-1/new_y2.shape[1]) * np.sum((new_delta_k2+new_model2.y)*np.log(new_model2.y))
    print('Gradient numerical: ' + str((loss1-loss2)/(2*(10**-2))))
    print('Gradient backprop: ' + str(backprop_vals['input_hidden_w_1_0']/batch_size))
    print('Difference: ' + str(np.abs(((loss1-loss2)/(2*(10**-2))) - \
        backprop_vals['input_hidden_w_1_0']/batch_size)))


    # compares this gradient with the math formula
    def get_approx(e, model_B, backprop_vals):
        # stores the difference of gradients
        diffs = []

        for layer in ["input_hidden", "hidden_output"]:
            for kind in ["weight", "bias"]:

                if kind == "weight":  # choose two weight values: w_00, w_10
                    for i in range(2):
                        model_plus_e = initialize_weight(
                            model_B, layer, kind, e, i, j=0)
                        model_minus_e = initialize_weight(
                            model_B, layer, kind, -e, i, j=0)

                        _, E0 = model_minus_e(x_train[:1], t_train[:1])
                        _, E1 = model_plus_e(x_train[:1], t_train[:1])
                        loss1 = (-1/model_plus_e.y.shape[1]) * np.sum((E1+model_plus_e.y)\
                            *np.log(model_plus_e.y))
                        loss0 = (-1/model_minus_e.y.shape[1]) * np.sum((E0+model_minus_e.y)\
                            *np.log(model_minus_e.y))

                        diff = np.abs((backprop_vals["_".join([layer, kind[0], 
                            str(i), str(0)])]/batch_size) - ((loss1 - loss0) / (2*e)))
                        diffs.append(diff)

                if kind == "bias":
                    model_plus_e = initialize_weight(
                        model_B, layer, kind, e, 0)
                    model_minus_e = initialize_weight(
                        model_B, layer, kind, -e, 0)

                    _, E0 = model_minus_e(x_train[:1], t_train[:1])
                    _, E1 = model_plus_e(x_train[:1], t_train[:1])
                    loss1 = (-1/model_plus_e.y.shape[1]) * np.sum((E1+model_plus_e.y)*np.log(model_plus_e.y))
                    loss0 = (-1/model_minus_e.y.shape[1]) * np.sum((E0+model_minus_e.y)*np.log(model_minus_e.y))

                    diff = np.abs((backprop_vals["_".join([layer, 
                        kind[0]])]/batch_size) - ((loss1 - loss0) / (2*e)))
                    diffs.append(diff)
        print(diffs)
        return (np.array(diffs) <= e**2).all()

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
        input_hidden_A.w = copy.deepcopy(input_hidden_B.w)
        hidden_output_A.w = copy.deepcopy(hidden_output_B.w)

        # reinitializes model biases to match the original model's
        input_hidden_A.b = copy.deepcopy(input_hidden_B.b)
        hidden_output_A.b = copy.deepcopy(hidden_output_B.b)

        # change value
        if kind == "weight":
            if layer == "input_hidden":
                input_hidden_A.w[i][j] += e
            else:
                hidden_output_A.w[i][j] += e
        else:
            if layer == "input_hidden":
                input_hidden_A.b[i][0] += e
            else:
                hidden_output_A.b[i][0] += e

        return model_A

    if get_approx(10**-2, model, backprop_vals):
        print("Success: gradients within range")
    else:
        print("Fail: gradients not within range")
'''
