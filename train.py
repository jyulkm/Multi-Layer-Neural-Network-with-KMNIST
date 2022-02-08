################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
from data import write_to_file, z_score_normalize
from neuralnet import *
import copy
import matplotlib.pyplot as plt


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
    
    # Defining terms for running mini-batch stochastic gradient descent with momentum
    batch_size = config['batch_size']
    epochs = config['epochs']
    early_stop_bool = config['early_stop']
    gamma = config['momentum_gamma']
    
    patience = 5 # number of epochs to wait till early stopping if validation error continues to go up
    
    # Keeping track of the recent loss for early stopping
    recent_loss = float('inf')

    lowest_loss = float('inf')

    # Checking if a regularization experiment is being run. Defines the model
    if experiment['type'] == 'regularization':
        model = NeuralNetwork(config=config, reg=True, reg_type=experiment['reg_type'])

    else:
        model = NeuralNetwork(config=config)
    
    # For early stopping with patience
    count = 0

    # Running through the epochs
    for e in range(epochs):

        # Iterating through each minbatch
        for minibatch in data.generate_minibatches(x_train, t_train, batch_size):
            x, t = minibatch

            y, deltas = model(x, t) # Forward call
        
            model.backward() # Backpropagating
        
        # Getting the training loss and accuracy of the model
        train_los, train_ac = test(model, x_train, t_train)
        train_acc.append(train_ac)
        train_loss.append(train_los)
        
        # Getting the validation loss and accuracy of the model
        val_los, val_ac = test(model, x_val, t_val)
        val_acc.append(val_ac)
        val_loss.append(val_los)

        # For getting the model that has the lowest loss on the validation data
        if val_los < lowest_loss:
            lowest_loss = val_los
            best_model = copy.deepcopy(model)

        # Checking validation loss for early stopping
        if val_los > recent_loss:
            count += 1
            if count == patience:
                break
        else:
            count = 0
            recent_loss = val_los

    return train_acc, val_acc, train_loss, val_loss, best_model
    
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
    y_labels, loss = model(x_test, t_test)
    accuracy = np.mean(np.argmax(y_labels, axis=1) == np.argmax(t_test, axis=1))

    return loss, accuracy

def train_mlp(x_train, t_train, x_val, t_val, x_test, t_test, config):
    """
    This function trains a single multi-layer perceptron and plots its performances.

    NOTE: For this function and any of the experiments, feel free to come up with your own ways of saving data
            (i.e. plots, performances, etc.). A recommendation is to save this function's data and each experiment's
            data into separate folders, but this part is up to you.
    """
    # Train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, t_train, x_val, t_val, config, experiment={'type': None})

    # Compute the loss and accuracy of the model on the test data
    test_loss, test_acc = test(best_model, x_test, t_test)

    # Building loss and accuracy plots
    plt.scatter(np.arange(len(train_loss)),np.array(train_loss) / x_train.shape[0],c='blue')
    plt.scatter(np.arange(len(valid_loss)),np.array(valid_loss) / x_val.shape[0],c='purple')
    plt.legend(['Training loss', 'Validation Loss'])
    plt.title('Baseline Model Loss vs. Number of Epochs')
    plt.ylabel('Baseline Model Loss')
    plt.xlabel('Number of Epochs')
    plt.show()

    plt.scatter(np.arange(len(train_acc)), train_acc, c='blue')
    plt.scatter(np.arange(len(valid_acc)), valid_acc, c='purple')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title('Baseline Model Accuracy vs. Number of Epochs')
    plt.ylabel('Baseline Model Accuracy')
    plt.xlabel('Number of Epochs')
    plt.show()

    # Printing out the average train/val loss and accuracy, and test loss and accuracy
    print("Config: %r" % config)
    print("Test Loss", test_loss / x_test.shape[0])
    print("Train Loss", np.mean(train_loss) / x_train.shape[0])
    print("Val Loss", np.mean(valid_loss) / x_val.shape[0])
    print("Test Accuracy", test_acc)
    print("Train Accuracy", np.mean(train_acc))
    print("Val Accuracy", np.mean(valid_acc))

    # The actual values
    train_loss = np.mean(train_loss) / x_train.shape[0]
    valid_loss = np.mean(valid_loss) / x_val.shape[0]
    test_loss = test_loss / x_test.shape[0]

    train_acc = np.mean(train_acc)
    valid_acc = np.mean(valid_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)


def activation_experiment(x_train, t_train, x_val, t_val, x_test, t_test, config):
    """
    This function tests all the different activation functions available and then plots their performances.
    """

    # Specify experiment details
    exp = {'type': None}

    # Train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, t_train, x_val, t_val, config, experiment=exp)

    # Check performance of model on test data
    test_loss, test_acc = test(best_model, x_test, t_test)

    # Building loss and accuracy plots
    plt.scatter(np.arange(len(train_loss)),np.array(train_loss) / x_train.shape[0],c='blue')
    plt.scatter(np.arange(len(valid_loss)),np.array(valid_loss) / x_val.shape[0],c='purple')
    plt.legend(['Training loss', 'Validation Loss'])
    plt.title(config['activation'].capitalize()+' Loss vs. Number of Epochs')
    plt.ylabel(config['activation'].capitalize()+' Loss')
    plt.xlabel('Number of Epochs')
    plt.show()

    plt.scatter(np.arange(len(train_acc)), train_acc, c='blue')
    plt.scatter(np.arange(len(valid_acc)), valid_acc, c='purple')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title(config['activation'].capitalize()+' Accuracy vs. Number of Epochs')
    plt.ylabel(config['activation'].capitalize()+' Accuracy')
    plt.xlabel('Number of Epochs')
    plt.show()

    # Printing out the average train/val loss and accuracy, and test loss and accuracy
    print("Config: %r" % config)
    print("Test Loss", test_loss / x_test.shape[0])
    print("Train Loss", np.mean(train_loss) / x_train.shape[0])
    print("Val Loss", np.mean(valid_loss) / x_val.shape[0])
    print("Test Accuracy", test_acc)
    print("Train Accuracy", np.mean(train_acc))
    print("Val Accuracy", np.mean(valid_acc))


def topology_experiment(x_train, t_train, x_val, t_val, x_test, t_test, config):
    """
    This function tests performance of various network topologies, i.e. making
    the graph narrower and wider by halving and doubling the number of hidden units.

    Then, we change number of hidden layers to 2 of equal size instead of 1, and keep
    number of parameters roughly equal to the number of parameters of the best performing
    model previously.
    """

    # Specify experiment details
    exp = {'type': None}

    # Train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, t_train, x_val, t_val, config, experiment=exp)

    # Check performance of model on test set
    test_loss, test_acc = test(best_model, x_test, t_test)

    # Building loss and accuracy plots
    plt.scatter(np.arange(len(train_loss)),np.array(train_loss) / x_train.shape[0],c='blue')
    plt.scatter(np.arange(len(valid_loss)),np.array(valid_loss) / x_val.shape[0],c='purple')
    plt.legend(['Training loss', 'Validation Loss'])
    plt.title('Loss vs. Number of Epochs (2 hidden layers)')
    plt.ylabel(config['activation'].capitalize()+' Loss')
    plt.xlabel('Number of Epochs')
    plt.show()

    plt.scatter(np.arange(len(train_acc)), train_acc, c='blue')
    plt.scatter(np.arange(len(valid_acc)), valid_acc, c='purple')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title('Accuracy vs. Number of Epochs (2 hidden layers)')
    plt.ylabel(config['activation'].capitalize()+' Accuracy')
    plt.xlabel('Number of Epochs')
    plt.show()

    # Printing out the average train/val loss and accuracy, and test loss and accuracy
    print("Config: %r" % config)
    print("Test Loss", test_loss / x_test.shape[0])
    print("Train Loss", np.mean(train_loss) / x_train.shape[0])
    print("Val Loss", np.mean(valid_loss) / x_val.shape[0])
    print("Test Accuracy", test_acc)
    print("Train Accuracy", np.mean(train_acc))
    print("Val Accuracy", np.mean(valid_acc))


def regularization_experiment(x_train, t_train, x_val, t_val, x_test, t_test, config):
    """
    This function tests the neural network with regularization.
    """

    # Specify experiment details
    exp = {'type': 'regularization', 'reg_type': 'L2'}

    # Train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, t_train, x_val, t_val, config, experiment=exp)

    # Check the performance of the model on the test set
    test_loss, test_acc = test(best_model, x_test, t_test)

    # Building loss and accuracy plots
    plt.scatter(np.arange(len(train_loss)),np.array(train_loss) / x_train.shape[0],c='blue')
    plt.scatter(np.arange(len(valid_loss)),np.array(valid_loss) / x_val.shape[0],c='purple')
    plt.legend(['Training loss', 'Validation Loss'])
    plt.title('L1 Loss vs. Number of Epochs')
    plt.ylabel('L1 Loss')
    plt.xlabel('Number of Epochs')
    plt.show()

    plt.scatter(np.arange(len(train_acc)), train_acc, c='blue')
    plt.scatter(np.arange(len(valid_acc)), valid_acc, c='purple')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title('L1 Accuracy vs. Number of Epochs')
    plt.ylabel('L1 Accuracy')
    plt.xlabel('Number of Epochs')
    plt.show()

    # Printing out the average train/val loss and accuracy, and test loss and accuracy
    print("Config: %r" % config)
    print("Test Loss", test_loss / x_test.shape[0])
    print("Train Loss", np.mean(train_loss) / x_train.shape[0])
    print("Val Loss", np.mean(valid_loss) / x_val.shape[0])
    print("Test Accuracy", test_acc)
    print("Train Accuracy", np.mean(train_acc))
    print("Val Accuracy", np.mean(valid_acc))


def update_weight(model, layer, kind, e=0, i=0, j=0):
    """
    Creates model_A and sets its weights and biases to match those of model_B:
    model_A.w = model_B.w + e
    """

    layer_map = {
        "input_to_hidden": 0,
        "hidden_to_output": 2
    }

    # Incrementing the weight/bias
    if kind == "weight":
        model.layers[layer_map[layer]].w[i][j] += e

    if kind == "bias":
        model.layers[layer_map[layer]].b[0][i] += e


def find_grad_diff(config, layer, kind, x_train, t_train, e, i=0, j=0):
    """
    Finds the difference between the gradient found by backpropagation
    and the one found numerically.
    """

    # Initializes the model
    model = NeuralNetwork(config=config)

    # Incrementing the weight/bias terms
    update_weight(model, layer, kind, e=e, i=i, j=j)
    _, loss1 = model(x_train, t_train)

    update_weight(model, layer, kind, e=-2*e, i=i, j=j)
    _, loss2 = model(x_train, t_train)
    approx_grad = abs((loss1 - loss2) / (2*e))

    update_weight(model, layer, kind, e=e, i=i, j=j)

    # Calling forward and backpropagating through the network
    model(x_train, t_train)
    model.backward()

    layer_map = {
        "input_to_hidden": 0,
        "hidden_to_output": 2
    }

    # Retrieve gradients calculated through backpropagation
    if kind == "weight":
        actual_grad = abs(model.layers[layer_map[layer]].d_w[i][j])
    if kind == "bias":
        actual_grad = abs(model.layers[layer_map[layer]].d_b[0])

    # Printing out the gradient details
    print(layer, kind, i, j)
    print("approximated gradient: ", approx_grad)
    print("back propagation gradient: ", actual_grad)
    print(np.isclose(approx_grad, actual_grad, e**2))
    print()


def check_gradients(x_train, t_train, config):
    """
    Check the network gradients computed by back propagation by 
    comparing with the gradients computed using numerical
    approximation.
    """

    # Increment value
    e = 10**-2

    # Using a single example to 
    x_train = np.array([x_train[0]])
    t_train = np.array([t_train[0]])

    # Finding the gradients for 2 weights and 1 bias in each layer
    for layer in ["input_to_hidden", "hidden_to_output"]:
        for kind in ["weight", "bias"]:
            if kind == "weight":
                for i in [0, 1]:
                    find_grad_diff(config, layer, kind, x_train, t_train, e, i)
            if kind == "bias":
                find_grad_diff(config, layer, kind, x_train, t_train, e)

