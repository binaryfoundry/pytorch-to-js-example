import sys
import random
import math
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from matplotlib import pyplot

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

def train(model, loader, optimizer, loss_fn, epochs):
    losses = []
    batch_index = 0
    for e in range(epochs):
        for in_data, out_data in loader:
            predict = model.forward(in_data)
            loss = loss_fn(predict, out_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            batch_index += 1
        print(f"epoch: {e+1}")
    return losses

def test(model, loader):
    predictions = []
    batch_index = 0
    for in_data, out_data in loader:
        predictions.append(model.forward(in_data).data.numpy())
        batch_index += 1
    return np.concatenate(predictions)

def plot_loss(losses):
    fig = pyplot.gcf()
    ax = pyplot.axes()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    x = list(range(len(losses)))
    pyplot.plot(x, losses)
    pyplot.show()
    pyplot.close()

def plot_predictions(test_data, predictions):
    fig = pyplot.figure()
    pyplot.scatter(test_data[:,0], test_data[:,1], marker='o', s=0.2)
    pyplot.scatter(predictions[:,0], predictions[:,1], marker='o', s=0.3)
    pyplot.text(-9, 0.44, "- Prediction", color="orange", fontsize=8)
    pyplot.text(-9, 0.48, "- Test Data", color="blue", fontsize=8)
    pyplot.show()

def js_write_layer(out_layer, layer_id=1, fnc=""):
    out_weights = out_layer.weight.data
    out_biases = out_layer.bias.data
    out_count = len(out_biases)
    output = ""
    for i in range(out_count):
        w = out_weights[i].tolist()
        b = out_biases[i].flatten().tolist()
        s = []
        for j in range(len(w)):
            s.append(f"{w[j]} * i{layer_id-1}_{j}")
        ww = " + ".join(s)
        output += f"var i{layer_id}_{i} = {fnc}(({ww}) + {b});\n"
    return output

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
epochs = 2

# hyper-parameter
learning_rate = 1e-3

# choose optimization scheme
model = Model()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

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

# train and test model
losses = train(model=model, loader=data_loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs)
predictions = test(model=model, loader=data_loader_test)

print("final loss:", sum(losses[-100:]) / 100)

plot_loss(losses)
plot_predictions(out_data_test, predictions)

# write model to js function
file = open("model.js","w+")
file.write("var tanh = Math.tanh;\n")
file.write("function evalModel(i0_0) {\n")
file.write(js_write_layer(model.layer0, 1, "tanh"))
file.write(js_write_layer(model.layer1, 2, "tanh"))
file.write(js_write_layer(model.layer2, 3, "tanh"))
file.write(js_write_layer(model.layer3, 4))
file.write("return [i4_0, i4_1];\n")
file.write("}\n")
file.close()
