import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from matplotlib import pyplot as plt

# Extracting the data
if os.path.exists("data/dataset.txt") is False:
    raise Exception("We don't have any training data. It should be stored as dataset.txt in the folder data.")

with open("data/dataset.txt", "r", encoding="utf-8") as f:
    data_raw = f.read().strip().split("\n\n")

data = [
        [[float(value) for value in line.split(',')] for line in mat_str.strip().split('\n')]
        for mat_str in data_raw
    ]
data = np.array(data)
input_data = data[:,1:-2,:]
target = data[:,2:-1,:]
train_data, test_data, train_target, test_target = train_test_split(input_data, target, test_size=0.2, random_state=42)

reshaped_train_data = torch.tensor(train_data[:,:,1:].reshape((train_data[:,:,1:].shape[0] * train_data[:,:,1:].shape[1], train_data[:,:,1:].shape[2])))
reshaped_train_target = torch.tensor(train_target[:,:,1:].reshape((train_target[:,:,1:].shape[0] * train_target[:,:,1:].shape[1], train_target[:,:,1:].shape[2])))
reshaped_test_data = torch.tensor(test_data[:,:,1:].reshape((test_data[:,:,1:].shape[0] * test_data[:,:,1:].shape[1], test_data[:,:,1:].shape[2])))
reshaped_test_target = torch.tensor(test_target[:,:,1:].reshape((test_target[:,:,1:].shape[0] * test_target[:,:,1:].shape[1], test_target[:,:,1:].shape[2])))

# Adding some arguments
parser = argparse.ArgumentParser(prog="learn_and_test.py",
                                 description="A pytorch code for learning and testing phase space\
                                 trajectory prediciton.")

parser.add_argument("--epochs", default=1000, type=int, help="number of epoches for the model to train")
parser.add_argument("--plot", default=True, type=bool, help="option of plotting the loss function")
args = parser.parse_args()

# Defining the neural network
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_device(DEVICE)

class TrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 30).double(),
            nn.Softplus(),
            nn.Linear(30, 30).double(),
            nn.Softplus(),
            nn.Linear(30, 2).double()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_stack(x)

network = TrajectoryModel()
L = nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr=1e-3)

#Training
losses = []
for _ in range(args.epochs):
    optimizer.zero_grad()
    loss = L(network(reshaped_train_data), reshaped_train_target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

if os.path.exists("models"):
    torch.save(network.state_dict(), "models/model.pth")
else:
    os.mkdir("models")
    torch.save(network.state_dict(), "models/model.pth")

MSE_test_set = L(network(reshaped_test_data), reshaped_test_target)
print(f"MSE on the test set is: {MSE_test_set}")

if args.plot == True:
    """plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("training loss decline on the training data")
    plt.plot(losses)"""

    if torch.numel(reshaped_test_data[0]) == 2:
        sample = np.array(test_data[np.random.randint(0,len(test_data)-1)])
        plt.xlabel("t")
        plt.ylabel("x")
        plt.plot(sample[:,0], sample[:,1], label="original data")
        prediction = [sample[0,1:]]
        for i in range(len(sample)):
            prediction.append(network(torch.tensor([prediction[i]]))[0].detach().numpy())
        prediction = np.array(prediction)
        plt.title(f"MSE: {MSE_test_set}")
        plt.plot(sample[:-2,0], prediction[:-3,0] , label="prediction")
        #plt.plot(sample[:-2,0], network(torch.from_numpy(sample[:-2,1:])).detach().numpy()[:,0] , label="prediction")

    plt.legend()
    plt.show()
