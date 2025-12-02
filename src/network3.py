import gzip
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def load_data_shared(filename="../data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        tr, va, te = pickle.load(f, encoding="latin1")

    def shared(data):
        x = torch.tensor(data[0], dtype=torch.float32)
        y = torch.tensor(data[1], dtype=torch.int64)
        return x, y

    return [shared(tr), shared(va), shared(te)]

class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def forward(self, x, training=False):
        if x.dim() == 2:
            x= x.view(x.size(0), 1 , 28, 28)
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        train_x, train_y = training_data
        val_x, val_y = validation_data
        test_x, test_y = test_data
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=mini_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_x, val_y),
            batch_size=mini_batch_size
        )
        test_loader = DataLoader(
            TensorDataset(test_x, test_y),
            batch_size=mini_batch_size
        )
        optimizer = torch.optim.SGD(self.params, lr=eta, weight_decay=lmbda)
        loss_fn = nn.CrossEntropyLoss()

        best_val = 0

        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                out = self.forward(x, training=True)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

                if batch_idx % 1000 == 0:
                    print(f"Epoch {epoch}, batch {batch_idx}, loss={loss.item():.4f}")

            # validação
            val_acc = self.accuracy_loader(val_loader)
            print(f"Epoch {epoch}: validation accuracy: {val_acc*100:.2f}%")

            if val_acc > best_val:
                print("Best result until now.")
                best_val = val_acc
                test_acc = self.accuracy_loader(test_loader)
                print(f"Corresponding test accuracy: {test_acc*100:.2f}%")

    def accuracy_loader(self, dataloader):
        correct = 0
        total = 0
        for x, y in dataloader:
            out = self.forward(x, training=False)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return correct / total

class ConvPoolLayer(object):
    def __init__(self, filter_shape, image_shape, poolsize=(2,2), activation_fn=torch.sigmoid):
        out_channels, in_channels, kernel_h, kernel_w = filter_shape
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_h, kernel_w))
        self.pool = nn.MaxPool2d(poolsize)
        self.activation_fn = activation_fn
        self.params = list(self.conv.parameters())

    def forward(self, x, training=False):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x

class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=torch.sigmoid, p_dropout=0.0):
        self.linear = nn.Linear(n_in, n_out)
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.dropout = nn.Dropout(p_dropout)
        self.params = list(self.linear.parameters())

    def forward(self, x, training=False):
        x = x.view(x.size(0), -1)
        if training and self.p_dropout > 0:
            x = self.dropout(x)
        x = F.relu(self.linear(x))
        return x

class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p_dropout)
        self.params = list(self.linear.parameters())

    def forward(self, x, training=False):
        x = x.view(x.size(0), -1)
        if training and self.dropout.p > 0:
            x = self.dropout(x)
        return self.linear(x)

    def accuracy(self, y):
        pass