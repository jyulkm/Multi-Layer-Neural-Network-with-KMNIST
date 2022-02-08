# CSE151B PA2

### Details on each Python file

The data.py file contains helper functions for one hot encoding, normalizing, and generating the minibatches we use in performing mini-batch stochastic gradient descent.

The neuralnet.py file contains the Activation, Layer, and Network classes that help build the neural network itself.

The train.py file is where we actually train and test the model. The 'train' function runs mini-batch stochastic gradient descent through our network with the specificied configurations. The 'test' function simply returns the accuracy and loss of our model on the input data. The 'train_mlp' function will call the 'train' function to train the model, test the best model on the test data, and print the resulting details on the loss and accuracy on each set of data. There is also code in the function for plotting the losses and accuracies. The functions 'activation_experiment', 'topology_experiment', and 'regularization_experiment' run their respective experiments on the network and plot the losses and accuracies. Finally, the 'check_gradients' function and its two helper functions ensure that the gradients we calculate during backpropagation are correct by comparing them to numerically calculated gradients.

Lastly, the main.py file takes in arguments from the command line. Available arguments:
  - '--train_mlp': Trains a single multi-layer perceptron using configs provided in config.yaml.
  - '--check_gradients': Checks the network gradients computed by comparing the gradient computed using.
  - '--regularization': Experiments with weight decay added to the update rule during training.
  - '--activation': Experimenting with different activation functions for hidden units.
  - '--topology': Experimenting with different network topologies.

The main.py file is also where we load in the data, split the data into train/test/validaiton, normalize the inputs, and one-hot encode the targets. It will then run the network based on the arguments passed in the command line.

> **_Example command:_** 'python main.py --train_mlp'.

> **_NOTE:_** For regularization, it will run L2. If you want it to do L1 regularization, add 'L1_penalty: [insert penalty]' to the config,yaml file and change the value in the dictionary at the top of the 'regularization_experiment' function in train.py from 'L2' to 'L1'.
