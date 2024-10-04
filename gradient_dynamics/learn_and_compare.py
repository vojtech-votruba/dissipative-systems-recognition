# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from matplotlib import pyplot as plt

# Extracting the data
if os.path.exists("data/dataset.txt") is False:
    raise Exception("We don't have any training data. It should be stored as dataset.txt in the folder data.")

with open("data/dataset.txt", "r", encoding="utf-8") as f:
    data_raw = f.read().strip().split("\n\n")

class TrajectoryDataset(Dataset):
        def __init__(self):
                loaded_trajectories = [
                        [[float(value) for value in line.split(',')] for line in mat_str.strip().split('\n')]
                        for mat_str in data_raw
                    ]
                loaded_trajectories = np.array(loaded_trajectories)
                data = loaded_trajectories[:,1:-2,:]
                target = loaded_trajectories[:,2:-1,:]

                global DIMENSION
                DIMENSION = int((len(data[0,0,:])-1)/2)

                self.data = torch.tensor(data[:,:,1:DIMENSION+1], requires_grad=True)
                self.target = torch.tensor(target[:,:,1:DIMENSION+1], requires_grad=True)

                """self.data = torch.tensor(data[:,:,1:DIMENSION+1].reshape(
                    (data[:,:,1:DIMENSION+1].shape[0] * data[:,:,1:DIMENSION+1].shape[1], data[:,:,1:DIMENSION+1].shape[2])), requires_grad=True)
                
                self.target = torch.tensor(target[:,:,1:DIMENSION+1].reshape(
                    (target[:,:,1:DIMENSION+1].shape[0] * target[:,:,1:DIMENSION+1].shape[1], target[:,:,1:DIMENSION+1].shape[2])), requires_grad=True)"""
                
                self.n_samples = self.data.shape[0]

        def __getitem__(self, index):
            return self.data[index], self.target[index]

        def __len__(self):
            return self.n_samples

trajectories = TrajectoryDataset()

"""
Older implementation

reshaped_train_data = torch.tensor(train_data[:,:,1:].reshape((train_data[:,:,1:].shape[0] * train_data[:,:,1:].shape[1], train_data[:,:,1:].shape[2])))
reshaped_train_target = torch.tensor(train_target[:,:,1:].reshape((train_target[:,:,1:].shape[0] * train_target[:,:,1:].shape[1], train_target[:,:,1:].shape[2])))
reshaped_test_data = torch.tensor(test_data[:,:,1:].reshape((test_data[:,:,1:].shape[0] * test_data[:,:,1:].shape[1], test_data[:,:,1:].shape[2])))
reshaped_test_target = torch.tensor(test_target[:,:,1:].reshape((test_target[:,:,1:].shape[0] * test_target[:,:,1:].shape[1], test_target[:,:,1:].shape[2])))

reshaped_test_data = torch.tensor(test_data[:,:,1:DIMENSION+1].reshape(
    (test_data[:,:,1:DIMENSION+1].shape[0] * test_data[:,:,1:DIMENSION+1].shape[1], test_data[:,:,1:DIMENSION+1].shape[2])), requires_grad=True)
reshaped_test_target = torch.tensor(test_target[:,:,1:DIMENSION+1].reshape(
    (test_target[:,:,1:DIMENSION+1].shape[0] * test_target[:,:,1:DIMENSION+1].shape[1], test_target[:,:,1:DIMENSION+1].shape[2])), requires_grad=True)
"""

# Adding some arguments
parser = argparse.ArgumentParser(prog="learn_and_test.py",
                                 description="A pytorch code for learning and testing phase space\
                                 trajectory prediciton.")

parser.add_argument("--epochs", default=200, type=int, help="number of epoches for the model to train")
parser.add_argument("--batch_size", default=50, type=int, help="batch size for training of the model")
parser.add_argument("--dt", default=0.02, type=float, help="size of the time step used in the simulation")
parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help="do you wish to train a new model?")
parser.add_argument('--plot', default=True, action=argparse.BooleanOptionalAction, help="option of plotting the loss function")
args = parser.parse_args()

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_device(DEVICE)
print(f"Using {DEVICE} for tensor calculations")

# Defining the neural network
class EntropyNetwork(nn.Module):
    """
        For the entropy network we are using a fully input concave neural network achitecture,
        originally designed by myself - it's a simple alteration of FICNN - fully input convex neural nets,
        we just need to use concave, decreasing activation functions and negative weights instead of positive ones.
    """
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(DIMENSION, 20).double()

        self.prop_layer1 = nn.Linear(20, 20).double()
        self.lateral_layer1 = nn.Linear(DIMENSION, 20).double()

        self.prop_layer2 = nn.Linear(20, 20).double()
        self.lateral_layer2 = nn.Linear(DIMENSION, 20).double()

        self.output_layer = nn.Linear(20, 1).double()
        self.lateral_layer_out = nn.Linear(DIMENSION, 1).double()

    def forward(self, x0):
        x = -nn.Softplus()(self.input_layer(x0))

        x = -nn.Softplus()(self.prop_layer1(x) + self.lateral_layer1(x0))
        self.prop_layer1.weight.data = -torch.abs(self.prop_layer1.weight.data)

        x = -nn.Softplus()(self.prop_layer2(x) + self.lateral_layer2(x0))
        self.prop_layer2.weight.data = -torch.abs(self.prop_layer2.weight.data)

        x_final = -nn.Softplus()(self.output_layer(x) + self.lateral_layer_out(x0))
        self.output_layer.weight.data = -torch.abs(self.output_layer.weight.data)
        
        return x_final

class DissipationNetwork(nn.Module):
    """
        For the dissipation potential network we are using a more complex architecture to ensure convexity of the output.
        Specifically: PICNN, source: https://arxiv.org/pdf/1609.07152
    """
    def __init__(self):
        super().__init__()
        # The branch that propagates x directly forward
        self.x_input_layer = nn.Linear(DIMENSION, 20).double()
        self.x_prop_layer1 = nn.Linear(20, 20).double()
        self.x_prop_layer2 = nn.Linear(20, 20).double()

        # The branch that goes directly between x and x_star
        self.x_lateral_layer_1 = nn.Linear(DIMENSION, 20).double()
        self.x_lateral_layer_2 = nn.Linear(20, 20).double()
        self.x_lateral_layer_3 = nn.Linear(20, 20).double()
        self.x_lateral_layer_out = nn.Linear(20, 1).double()

        # The branch that propagates x_star forward (We need to enforce convexity here)
        self.conjugate_prop_layer_1 = nn.Linear(20, 20, bias=False).double()
        self.conjugate_prop_layer_2 = nn.Linear(20, 20, bias=False).double()
        self.conjugate_prop_layer_out= nn.Linear(20, 1, bias=False).double()

        self.conjugate_prop_layer_1_mid = nn.Linear(20, 20).double()
        self.conjugate_prop_layer_2_mid = nn.Linear(20, 20).double()
        self.conjugate_prop_layer_out_mid = nn.Linear(20, 20).double()

        # The branch which always starts at x0_star and ends at arbitrary x_star
        self.conjugate_lateral_layer_in = nn.Linear(DIMENSION, 20, bias=False).double()
        self.conjugate_lateral_layer_1 = nn.Linear(DIMENSION, 20, bias=False).double()
        self.conjugate_lateral_layer_2 = nn.Linear(DIMENSION, 20, bias=False).double()
        self.conjugate_lateral_layer_out = nn.Linear(DIMENSION, 1, bias=False).double()

        self.conjugate_lateral_layer_in_mid = nn.Linear(DIMENSION, DIMENSION).double()
        self.conjugate_lateral_layer_1_mid = nn.Linear(20, DIMENSION).double()
        self.conjugate_lateral_layer_2_mid = nn.Linear(20, DIMENSION).double()
        self.conjugate_lateral_layer_out_mid = nn.Linear(20, DIMENSION).double()

    def forward(self, input):
        x0 = input[:,:int(input.size(1)/2)]
        x0_star = input[:,int(input.size(1)/2):]

        x_star = nn.Softplus()(self.x_lateral_layer_1(x0) + self.conjugate_lateral_layer_in(torch.mul(x0_star, self.conjugate_lateral_layer_in_mid(x0))))
        x = nn.Softplus()(self.x_input_layer(x0))

        x_star = nn.Softplus()(self.x_lateral_layer_2(x) + self.conjugate_prop_layer_1(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_1_mid(x))))
                                + self.conjugate_lateral_layer_1(torch.mul(x0_star, self.conjugate_lateral_layer_1_mid(x))))
        x = nn.Softplus()(self.x_prop_layer1(x))
        self.conjugate_prop_layer_1.weight.data = torch.abs(self.conjugate_prop_layer_1.weight.data)

        x_star = nn.Softplus()(self.x_lateral_layer_3(x) + self.conjugate_prop_layer_2(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_2_mid(x))))
                                + self.conjugate_lateral_layer_2(torch.mul(x0_star, self.conjugate_lateral_layer_2_mid(x))))
        x = nn.Softplus()(self.x_prop_layer2(x))
        self.conjugate_prop_layer_2.weight.data = torch.abs(self.conjugate_prop_layer_2.weight.data)

        out = nn.Softplus()(self.x_lateral_layer_out(x) + self.conjugate_prop_layer_out(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_out_mid(x))))\
                                + self.conjugate_lateral_layer_out(torch.mul(x0_star, self.conjugate_lateral_layer_out_mid(x))))
        self.conjugate_prop_layer_out.weight.data = torch.abs(self.conjugate_prop_layer_out.weight.data)

        return out

class GradientDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = EntropyNetwork()
        self.Xi = DissipationNetwork()

    def forward(self, x):
        S = self.S(x)
        x_star = autograd.grad(S, x, grad_outputs=torch.ones_like(S), create_graph=True)[0]
        input = torch.stack((x,x_star), dim=1)

        Xi = self.Xi(input)
        x_dot = autograd.grad(Xi, x_star, grad_outputs=torch.ones_like(Xi), create_graph=True)[0]

        return [Xi,x_dot]
    
    def dissipation(self,x_star_tensor):
        zeros_column = torch.zeros_like(x_star_tensor, dtype=torch.float64)
        input = torch.stack((zeros_column, x_star_tensor), dim=1)
        return self.Xi(input)

L = nn.MSELoss()

if args.train:
    training_trajectories, test_trajectories = random_split(trajectories, [0.8, 0.2])
    dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size, shuffle=True)

    model = GradientDynamics()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    losses = []
    for i in range(args.epochs):
        for j, (trajectory_batch_data, trajectory_batch_target) in enumerate(dataloader):
            optimizer.zero_grad()
            predictions = model(trajectory_batch_data)[1]
            loss = L(predictions * args.dt + trajectory_batch_data, trajectory_batch_target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch no. {i}/{args.epochs} done!")

    if os.path.exists("models"):
        torch.save(model.state_dict(), "models/model.pth")
    else:
        os.mkdir("models")
        torch.save(model.state_dict(), "models/model.pth")
else:
    model = GradientDynamics()
    model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    model.eval

if args.train:
    test_data = trajectories.data[test_trajectories.indices]
    test_target = trajectories.target[test_trajectories.indices]

else:
    test_data = trajectories.data
    test_target = trajectories.target

MSE_test_set = L(model(test_data)[1]*args.dt+test_data, test_target)
print(f"MSE on the test set is: {MSE_test_set}")

if args.plot:
    plt.style.use('ggplot')
    # Plotting the MSE decline
    if args.train:
        fig1,ax1 = plt.subplots()
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("MSE")
        ax1.set_title("training loss decline on the training data")
        ax1.plot(range(len(losses)), losses)

    if DIMENSION == 1:
        # Sampling random trajectory and plotting it along with predicted trajectory
        fig2,ax2 = plt.subplots()
        sample = test_data[np.random.randint(0,len(test_data)-1)].detach().numpy()
        time = [args.dt*i for i in range(len(sample))]
    
        ax2.set_xlabel("t")
        ax2.set_ylabel("x")
        ax2.plot(time, sample, label="original data")

        prediction = [sample[0]]
        for i in range(len(sample)):
            prediction.append(float(model(torch.tensor([prediction[i]], requires_grad=True).unsqueeze(0))[1])*args.dt+prediction[i])

        prediction = np.array(prediction)
        ax2.set_title(f"MSE of the test set: {MSE_test_set}")
        ax2.plot(time[:-2], prediction[:-3] , label="prediction")
        ax2.legend()

        # Plotting dissipation potential
        fig3,ax3 = plt.subplots()
        ax3.set_xlabel("x*")
        ax3.set_ylabel("Ξ")
        x_star_range = torch.linspace(-50,50,200)
        ax3.plot(x_star_range, model.dissipation(x_star_range).detach())
        ax3.set_title("Dissipation potential Ξ = Ξ(x*)")

        # Plotting entropy
        fig4,ax4 = plt.subplots()
        ax4.set_xlabel("x")
        ax4.set_ylabel("S")
        x = torch.linspace(-50,50,500, dtype=torch.float64)
        x = x.view(-1, DIMENSION)
        ax4.plot(x, model.S(x).detach())
        ax4.set_title("Entropy S = S(x)")
        
    plt.show()
    