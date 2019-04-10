import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim as ls_schedule
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm

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
            self.activation_function2 = nn.ReLU()
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

        out = self.hidden2(out)
        out = self.activation_function2(out)
        if self.use_bn:
            out = self.hidden2_bn(out)

        out = self.hidden3(out)
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
    fig, axs = plt.subplots(2, 3, tight_layout=True)

    h1_w = model.hidden1.weight.detach().numpy().flatten()
    h1_b = model.hidden1.bias.detach().numpy().flatten()

    h2_w = model.hidden2.weight.detach().numpy().flatten()
    h2_b = model.hidden2.bias.detach().numpy().flatten()

    h3_w = model.hidden3.weight.detach().numpy().flatten()
    h3_b = model.hidden3.bias.detach().numpy().flatten()

    axs[0, 0].hist(h1_w)
    axs[0, 1].hist(h2_w)
    axs[0, 2].hist(h3_w)
    axs[1, 0].hist(h1_b)
    axs[1, 1].hist(h2_b)
    axs[1, 2].hist(h3_b)

    axs[0, 0].set_title('hidden1_weight')
    axs[0, 1].set_title('hidden2_weight')
    axs[0, 2].set_title('hidden3_weight')
    axs[1, 0].set_title('hidden1_bias')
    axs[1, 1].set_title('hidden2_bias')
    axs[1, 2].set_title('hidden3_bias')
    fig.savefig('fig.png')

def weight_bias_reset(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            mean,std = 0, 0.1
            torch.nn.init.normal_(m.weight, mean, std)
            torch.nn.init.normal_(m.bias, mean, std)

def weight_bias_reset_constant(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 1)

def weight_bias_reset_xavier_uniform(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.constant_(m.bias, 1)

def weight_bias_reset_kaiming_uniform(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.constant_(m.bias, 1)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(train_loader, model, loss_fn, optimizer, get_grad=False):
    model.train()
    total_loss = 0
    grad_2 = 0.0
    grad_3 = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, target)
        total_loss += loss.item()
        loss.backward()

        if get_grad:
            g2, g3 = model.get_grad()
            grad_2 += g2
            grad_3 += g3

        optimizer.step()

    average_loss = total_loss / batch_idx
    average_grad2 = grad_2 / batch_idx
    average_grad3 = grad_3 / batch_idx
    return average_loss, average_grad2, average_grad3

def evaluate(loader, model, loss_fn):
    with torch.no_grad():
        model.eval()
        correct = 0.0
        total_loss = 0

        for batch_idx, (data, target) in enumerate(loader):
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().detach().numpy()
            loss = loss_fn(outputs, target)
            total_loss += loss.item()
        accuracy = correct * 100.0 / len(loader.dataset)
    return total_loss, accuracy

def fit(train_loader, val_loader, model, loss_fn, optimizer, n_epoches, get_grad=False):
    grad_2 = []
    grad_3 = []

    train_access = []
    train_losses = []

    for epoch in range(n_epochs):
        train_loss, average_grad2, average_grad3 = train(train_loader, model, loss_fn, optimizer, get_grad)

        _, train_accuracy = evaluate(train_loader, model, loss_fn)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, n_epochs, train_loss, train_accuracy)
        print(message)

        train_access.append(train_accuracy)
        train_losses.append(train_loss)
        grad_2.append(average_grad2)
        grad_3.append(average_grad3)

        val_loss, val_accuracy = evaluate(val_loader, model, loss_fn)
        message = 'Epoch: {}/{}. Validation set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, n_epochs, val_loss, val_accuracy)
        print(message)

    if get_grad:
        fig, ax = plt.subplots()
        ax.plot(grad_2, label='Gradient for Hidden 2 Layer')
        ax.plot(grad_3, label='Gradient for Hidden 3 Layer')
        plt.ylim(top=0.004)
        legend = ax.legend(loc='best', shadow=True, fontize='x-large')
    return train_access, train_losses

def show_curve(ys, title):
    x = np.array(range(len(ys)))
    y = np.array(ys)
    plt.plot(x, y, c='b')
    plt.axis()
    plt.title('{} Curve:'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('{} Value'.format(title))
    plt.show()

if __name__ == "__main__":
    cuda = torch.device('cuda:0')

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
    loss_fn = torch.nn.CrossEntropyLoss()
    # l2 norm can be done in SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_norm)

    # weight_bias_reset(model)
    # weight_bias_reset_constant(model)
    # weight_bias_reset_xavier_uniform(model)
    # weight_bias_reset_kaiming_uniform(model)
    # show_weight_bias(model)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=test_transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))

    train_accs, train_losses = fit(train_loader, test_loader, model, loss_fn, optimizer, n_epochs, get_grad)