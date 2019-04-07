import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim import ls_schedule
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdn

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function='RELU'):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.use_dropout = False
        self.use_bn = False
        # 784 ---> 100
        self.hidden1 = nn.Linear(input_size, hidden_size)
        # 100 ---> 100
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        # 100 ---> 100
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        # 100 ---> 10
        self.classification_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)
        self.hidden1_bn = nn.BatchNorm1d(hidden_size)
        self.hidden2_bn = nn.BatchNorm1d(hidden_size)
        self.hidden3_bn = nn.BatchNorm1d(hidden_size)

        if activation_function == 'SIGMOID':
            self.activation_function1 = nn.Sigmoid()
            self.activation_function2 = nn.Sigmoid()
            self.activation_function3 = nn.Sigmoid()
        else:
            self.activation_function1 = nn.ReLU()
            self.activation_function3 = nn.ReLU()
            self.activation_function3 = nn.ReLU()

    def forward(self, x):
        '''
        Defines the computation performed at every cell
        Should be overwritten by all subclasses.
        Args:
            x: [batch_size, channel, height, width], input for network
        Return:
            out: [batch_size, n_classes], output from network
        '''
        x = x.view(x.size(0), -1)
        out = self.hidden1(x)
        out = self.activation_function1(out)
        if self.use_bn:
            out = self.hidden1_bn(out)

        out = self.hidden2(x)
        out = self.activation_function2(out)
        if self.use_bn:
            out = self.hidden2_bn(out)

        out = self.hidden3(x)
        if self.use_bn:
            out = self.hidden1_bn(out)
        out = self.activation_function3(out)

        if self.use_dropout:
            out = self.dropout(out)

        out = self.classification_layer(out)
        return out

    def set_use_dropout(self, use_dropout):
        self.use_dropout = use_dropout

    def set_use_bn(self, use_bn):
        self.use_bn = use_bn

    def get_grad(self):
        hidden2_average_grad = np.mean(np.sqrt(self.hidden2.weight.grad.detach().numpy()))
        hidden3_average_grad = np.mean(np.sqrt(self.hidden3.weight.grad.detach().numpy()))
        return hidden2_average_grad, hidden3_average_grad


def show_weight_bias(model):
    fig, axs = plt.subplots(2, 3, sharey=False, tight_layout=True)

    h1_w = model.hidden1.weight.detach().numpy().flatten()
    h1_b = model.hidden1.bias.detach().numpy().flatten()

    h2_w = model.hidden2.weight.detach().numpy().flatten()
    h2_b = model.hidden2.bias.detach().numpy().flatten()

    h3_w = model.hidden3.weight.detach().numpy().flatten()
    h3_b = model.hidden3.bias.detach().numpy().flatten()

    ax[0, 0].hist(h1_w)
    ax[0, 1].hist(h2_w)
    ax[0, 2].hist(h3_w)
    ax[1, 0].hist(h1_b)
    ax[1, 1].hist(h2_b)
    ax[1, 2].hist(h3_b)

def weight_bias_reset(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            mean,std = 0, 0.1
        torch.nn.init.normal_(m.weight, mean, std)
        torch.nn.init.normal_(m.bias, mean, std)

def weight_bias_reset_constant(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            pass

def weight_bias_reset_xavier_uniform(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            pass

def weight_bias_reset_kaiming_uniform(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            pass

if __name__ == "__main__":
    cuda = torch.device('cpu')

    batch_size = 128
    n_epochs = 5
    learning_rate = 0.01
    input_size = 28 * 28
    hidden_size = 100
    output_size = 10
    l2_norm = 0
    dropout = False
    get_grad = False

    model = FeedForwardNeuralNetwork(input_size, hidden_size, output_size)
    # Cross entropy
    loss_fn = torch.nnCrossEntropyLoss()
    # l2 norm can be done in SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_norm)

    weight_bias_reset(model)
    show_weight_bias(model)