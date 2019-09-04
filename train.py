import random
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        in_count = 1  # only one variable as input 'x'.
        hidden_1 = 64 # hidden layers have 64 nodes each.
        hidden_2 = 64
        hidden_3 = 64
        out_count = 2 # two output vars 'u' and 'v'.

        self.layer0 = nn.Linear(in_count, hidden_1)
        self.layer1 = nn.Linear(hidden_1, hidden_2)
        self.layer2 = nn.Linear(hidden_2, hidden_3)
        self.layer3 = nn.Linear(hidden_3, out_count)

    def forward(self, x):
        x = self.layer0(x)
        x = torch.tanh(x)
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        return x

class Dataset(Dataset):
    def __init__(self, in_data, out_data):
        self.length = in_data.shape[0]
        self.in_data = torch.from_numpy(in_data).type(torch.FloatTensor)
        self.out_data = torch.from_numpy(out_data).type(torch.FloatTensor)

    def __getitem__(self, index):
        return self.in_data[index], self.out_data[index]

    def __len__(self):
        return self.length

# random value 0 < x < 2Pi
def tau_rand(x):
    return random.uniform(0, 2 * math.pi)

# map function and convert to list
def lmap(func, data):
    return list(map(func, data))

# No. of data items for training and for testing
train_data_size = 100000
test_data_size = 8000

# No. of samples used for back propagation gradient
batch_size = 16

# create arrays of random values 0 < x < 2Pi
in_data_x_train = lmap(tau_rand, [0] * train_data_size)
in_data_x_test = lmap(tau_rand, [0] * test_data_size)

# sin and cos values to lean from training data
out_data_u_train = lmap(math.sin, in_data_x_train)
out_data_v_train = lmap(math.cos, in_data_x_train)

# sin and cos values to test learned model against
out_data_u_test = lmap(math.sin, in_data_x_test)
out_data_v_test = lmap(math.cos, in_data_x_test)

# convert data to numpy tables
in_data_train = np.row_stack(in_data_x_train)
in_data_test = np.row_stack(in_data_x_test)

out_data_train = np.column_stack((out_data_u_train, out_data_v_train))
out_data_test = np.column_stack((out_data_u_test, out_data_v_test))

# numpy tables to pytorch Datasets
data_train = Dataset(in_data=in_data_train, out_data=out_data_train)
data_test = Dataset(in_data=in_data_test, out_data=out_data_test)

data_loader_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(dataset=data_test, batch_size=len(data_test), shuffle=False)
