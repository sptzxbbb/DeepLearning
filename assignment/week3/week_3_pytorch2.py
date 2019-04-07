import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import csv
import time

class Network(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Network, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h1 = self.hidden(x)
        h2 = self.sigmoid(h1)
        out = self.predict(h2)
        return out

class PointDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, encoding='utf-8', header=None)
        print('csv_file source ----->', csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        x = self.frame.iloc[idx, 0]
        y = self.frame.iloc[idx, 1]
        point = np.array([x, y])
        label = int(self.frame.iloc[idx, 2])
        if self.transform is not None:
            point = self.transform(point)
        sample = {'point': point, 'label': label}
        return sample

def show_running_loss(running_loss):
    x = np.array([i for i in range(len(running_loss))])
    y = np.array(running_loss)
    plt.figure()
    plt.plot(x, y, c = 'b')
    plt.axis()
    plt.title('loss curve:')
    plt.xlabel('step')
    plt.ylabel('loss value')
    plt.show()

def show_accuracy(running_accuracy):
    x = np.array([i for i in range(len(running_accuracy))])
    y = np.array(running_accuracy)
    plt.figure()
    plt.plot(x, y, c = 'b')
    plt.axis()
    plt.title('accuracy curve')
    plt.xlabel('step')
    plt.ylabel('accuracy value')
    plt.show()

def show_original_points():
    label_csv = open('./labels/label.csv', 'r')
    label_writer = csv.reader(label_csv)
    class1_point = []
    class2_point = []
    class3_point = []
    for item in label_writer:
        k = [item[0], item[1]]
        if item[2] == '0':
            class1_point.append(k)
        elif item[2] == '1':
            class2_point.append(k)
        else:
            class3_point.append(k)
    data1 = np.array(class1_point, dtype=float)
    data2 = np.array(class2_point, dtype=float)
    data3 = np.array(class3_point, dtype=float)
    x1, y1 = data1.T
    x2, y2 = data2.T
    x3, y3 = data3.T
    plt.figure()
    plt.scatter(x1, y1, c='b', marker='.')
    plt.scatter(x2, y2, c='r', marker='.')
    plt.scatter(x3, y3, c='g', marker='.')
    plt.axis()
    plt.title('scatter')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def predict(classifier_net, testloader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            point, label = data['point'], data['label']
            point, label = point.to(device).to(torch.float32), label.to(device)
            outputs = classifier_net(point)
            _, predicted = torch.max(outputs, 1)
            print('model prediction: ', predicted)
            print('ground truth: ', label, '\n')
            correct += (predicted == label).sum()
            total += label.size(0)
            print('current correct is:', correct.item())
            print('current total is:', total)
        print('the accuracy of the model is {0:5f}'.format(correct.item()/total))

    return correct.item() / total

def train(classifier_net, trainloader, testloader, device, lr, optimizer):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optimizer
    running_loss = []
    running_accuracy = []
    for epoch in range(epoches):
        temp_loss = 0
        iteration = 0
        for i, data in enumerate(trainloader):
            point, label = data['point'], data['label']
            point, label = point.to(device).to(torch.float32), label.to(device)

            outputs = classifier_net(point)

            loss = criterion(outputs, label)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            temp_loss += loss.item()
            iteration += 1
        running_loss.append(temp_loss / iteration)
        print('test {}:-------------------------------------------------'.format(epoch))
        running_accuracy.append(predict(classifier_net, testloader, device))

    show_running_loss(running_loss)
    show_accuracy(running_accuracy)
    return classifier_net

if __name__ == "__main__":
    epoches = 100

    lr = 1e-2

    batch_size = 30

    transform = torch.tensor

    device = torch.device('cpu')

    trainset = PointDataset('./labels/label.csv', transform=transform)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = PointDataset('./labels/test.csv', transform=transform)

    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=4)

    # show_original_points()

    classifier_net = Network(2, 5, 3).to(device)

    optimizer1 = optim.SGD(classifier_net.parameters(), lr=lr, momentum=0)
    optimizer2 = optim.Adam(classifier_net.parameters(), lr=lr)
    optimizer3 = optim.Rprop(classifier_net.parameters(), lr=lr)
    classifier_net = train(classifier_net, trainloader, testloader, device, lr, optimizer3)



