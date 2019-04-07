import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

class FeedForward_Neural_Network(object):
    def __init__(self, learning_rate):
        self.input_channel = 2  #  number of input neurons
        self.output_channel = 3 #  number of output neurons
        self.hidden_channel = 5 # number of hidden neurons
        self.learning_rate = learning_rate

        # weights initialization
        # Usually, we use random or uniform initialzation to initialize weight
        # For simplicity, here we use same array to initialze
#         np.random.randn(self.input_channel, self.hidden_channel)
        # (2x5) weight matrix from input to hidden layer
        self.weight1 = np.array([[ 2.12444863,  0.25264613,  1.45417876,  0.56923979,  0.45822365],
                                 [-0.80933344,  0.86407349,  0.20170137, -1.87529904, -0.56850693]])

        # (5x3) weight matrix from hidden to output layer
#         np.random.randn(self.hidden_channel, self.output_channel)
        self.weight2 = np.array([ [-0.06510141,  0.80681666, -0.5778176 ],
                               [ 0.57306064, -0.33667496,  0.29700734],
                               [-0.37480416,  0.15510474,  0.70485719],
                               [ 0.8452178 , -0.65818079,  0.56810558],
                               [ 0.51538125, -0.61564998,  0.92611427]])


    def forward(self, X):
        """forward propagation through our network
        """
        # dot product of X (input) and first set of 3x2 weights
        self.h1 = np.dot(X, self.weight1)
        # activation function
        self.z1 = self.sigmoid(self.h1)
        # dot product of hidden layer (z2) and second set of 3x1 weights
        self.h2 = np.dot(self.z1, self.weight2)
        # final activation function
        o = self.sigmoid(self.h2)
        return o

    def backward(self, X, y, o):
        """Backward, compute gradient and update parameters
        Inputs:
            X: data, [batch_size, 2]
            y: label, one-hot vector, [batch_size, 3]
            o: predictions, [batch_size, 3]
        """
        # backward propgate through the network
        self.o_error = y - o  # error in output
         # applying derivative of sigmoid to error  delata L
        self.o_delta = self.o_error * self.sigmoid_prime(o)

        # z1 error: how much our hidden layer weights contributed to output error
        self.z1_error = self.o_delta.dot(self.weight2.T)
        # applying derivative of sigmoid to z1 error
        self.z1_delta = self.z1_error * self.sigmoid_prime(self.z1)

        # adjusting first set (input --> hidden) weights
        self.weight1 += X.T.dot(self.z1_delta) * self.learning_rate
        # adjusting second set (hidden --> output) weights
        self.weight2 += self.z1.T.dot(self.o_delta) * self.learning_rate

    def sigmoid(self, s):
        """activation function
        """
        return 1 / (1 + np.exp(-s))

    def sigmoid_prime(self, s):
        """derivative of sigmoid
        """
        return s * (1 - s)


def plot(data, labels, caption):
    """plot the data distribution, !!YOU CAN READ THIS LATER, if you are interested
    """
    colors = cm.rainbow(np.linspace(0, 1, len(set(labels))))
    for i in set(labels):
        xs = []
        ys = []
        for index, label in enumerate(labels):
            if label == i:
                xs.append(data[index][0])
                ys.append(data[index][1])
        plt.scatter(xs, ys, colors[int(i)])
    plt.title(caption)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def int2onehot(label):
    """conver labels into one-hot vector, !!YOU CAN READ THIS LATER, if you are interested
    Args:
        label: [batch_size]
    Returns:
        onehot: [batch_size, categories]
    """
    dims = len(set(label))
    imgs_size = len(label)
    onehot = np.zeros((imgs_size, dims))
    onehot[np.arange(imgs_size), label] = 1
    return onehot

def get_accuracy(predictions, labels):
    """Compute accuracy, !!YOU CAN READ THIS LATER, if you are interested
    Inputs:
        predictions:[batch_size, categories] one-hot vector
        labels: [batch_size, categories]
    """
    predictions = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    all_imgs = len(labels)
    predict_true = np.sum(predictions == labels)
    return predict_true/all_imgs

# Please read this function carefully, related to implementation of GD, SGD, and mini-batch
def generate_batch(train_data, train_labels, batch_size):
    """Generate batch
    when batch_size=len(train_data), it's GD
    when batch_size=1, it's SGD
    when batch_size>1 & batch_size<len(train_data), it's mini-batch, usually, batch_size=2,4,8,16...
    """
    iterations = math.ceil(len(train_data)/batch_size)
    for i in range(iterations):
        index_from = i*batch_size
        index_end = (i+1)*batch_size
        yield (train_data[index_from:index_end], train_labels[index_from:index_end])

def show_curve(ys, title):
    """plot curlve for Loss and Accuacy, !!YOU CAN READ THIS LATER, if you are interested
    Args:
        ys: loss or acc list
        title: Loss or Accuracy
    """
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('{} Value'.format(title))
    plt.show()


if __name__ == "__main__":
    train_csv_file = './labels/train.csv'
    test_csv_file = './labels/test.csv'
    # Load data from csv file, without header
    train_frame = pd.read_csv(train_csv_file, encoding='utf-8', header=None)
    test_frame = pd.read_csv(test_csv_file, encoding='utf-8', header=None)

    # obtain data from specific columns

    # obtain data from first and second columns and convert into narray
    train_data = train_frame.iloc[:,0:2].values
    # obtain labels from third columns and convert into narray
    train_labels = train_frame.iloc[:,2].values
    # obtain data from first and second columns and convert into narray
    test_data = test_frame.iloc[:,0:2].values
    # obtain labels from third columns and convert into narray
    test_labels = test_frame.iloc[:,2].values

    # train & test data shape
    print(train_data.shape)
    print(test_data.shape)
    # train & test labels shape
    print(train_labels.shape)
    print(test_labels.shape)

    plot(train_data, train_labels, 'train_dataset')

    plot(test_data, test_labels, 'test_dataset')

    # convert labels into one hot vector
    train_labels_onehot = int2onehot(train_labels)
    test_labels_onehot = int2onehot(test_labels)
    print(train_labels_onehot.shape)
    print(train_labels_onehot.shape)

    learning_rate = 0.01
    epochs = 400 # training epoch
    # batch_size = len(train_data) # GD
    # batch_size = 1               # SGD
    batch_size = 8               # mini-batch
    model = FeedForward_Neural_Network(learning_rate) # declare a simple feedforward neural model

    losses = []
    accuracies = []

    for i in range(epochs):
        loss = 0
        for index, (xs, ys) in enumerate(generate_batch(train_data, train_labels_onehot, batch_size)):
            predictions = model.forward(xs) # forward phase
            loss += 1/2 * np.mean(np.sum(np.square(ys-predictions), axis=1)) # Mean square error
            model.backward(xs, ys, predictions) # backward phase

        losses.append(loss)

        # train dataset acc computation
        predictions = model.forward(train_data)
        # compute acc on train dataset
        accuracy = get_accuracy(predictions, train_labels_onehot)
        accuracies.append(accuracy)

        if i % 50 == 0:
            print('Epoch: {}, has {} iterations'.format(i, index+1))
            print('\tLoss: {:.4f}, \tAccuracy: {:.4f}'.format(loss, accuracy))

    test_predictions = model.forward(test_data)
    # compute acc on test dataset
    test_accuracy = get_accuracy(test_predictions, test_labels_onehot)
    print('Test Accuracy: {:.4f}'.format(test_accuracy))

    show_curve(losses, 'loss')
    show_curve(accuracies, 'accuracy')