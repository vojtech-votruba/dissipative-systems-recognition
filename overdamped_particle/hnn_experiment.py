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

                self.position = torch.tensor(data[:,:,1:DIMENSION+1], requires_grad=True).float()
                self.velocity = torch.tensor(data[:,:,DIMENSION+1:], requires_grad=True).float()
                self.target_pos = torch.tensor(target[:,:,1:DIMENSION+1], requires_grad=True).float()
                self.target_vel = torch.tensor(target[:,:,DIMENSION+1:], requires_grad=True).float()
                
                self.n_samples = self.position.shape[0]

        def __getitem__(self, index):
            return self.position[index], self.velocity[index], self.target_pos[index], self.target_vel[index]

        def __len__(self):
            return self.n_samples

trajectories = TrajectoryDataset()

# Adding some arguments
parser = argparse.ArgumentParser(prog="learn_and_test.py",
                                 description="A pytorch code for learning and testing phase space\
                                 trajectory prediciton.")

parser.add_argument("--epochs", default=100, type=int, help="number of epoches for the model to train")
parser.add_argument("--batch_size", default=50, type=int, help="batch size for training of the model")
parser.add_argument("--dt", default=0.01, type=float, help="size of the time step used in the simulation")
parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help="do you wish to train a new model?")
parser.add_argument('--plot', default=True, action=argparse.BooleanOptionalAction, help="option of plotting the loss function")
parser.add_argument("--log", default=True, type=int, help="using log loss for plotting and such")
args = parser.parse_args()

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# My GPU doesn't have enough memory to handle these computations
DEVICE = "cpu"

torch.set_default_device(DEVICE)
print(f"Using {DEVICE} for tensor calculations")

def enforce_pos(M):
    """
        This doesn't have anything to do with positive reinforcement,
        we just need a convinient way to enforce that all of the elements
        of a tensor are going to be positive
    """
    return torch.abs(M)

def rk4(f, x, time_step):
    """
        Classical 4th order Runge Kutta implementation,
        currently not used.
    """
    k1i = f(x)
    k2i = f(x + k1i * time_step/2)
    k3i = f(x + k2i * time_step/2)
    k4i = f(x + k3i * time_step)

    return 1/6 * (k1i + 2*k2i + 2*k3i + k4i)

def hamiltonian_model(DIMENSION):
    """
    Simple feedforward neural network to approximate the Hamiltonian.
    """
    return nn.Sequential(
        nn.Linear(2 * DIMENSION, 64),
        nn.Softplus(),
        nn.Linear(64, 64),
        nn.Softplus(),
        nn.Linear(64, 1)
    )

class HamiltonianNN(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.hamiltonian = hamiltonian_model(dimension)
    
    def forward(self, x, p):
        # Concatenating x and p to feed into the Hamiltonian model
        state = torch.cat((x, p), dim=-1)
        return self.hamiltonian(state)

def compute_time_derivatives(model, x):
    """
    Computes dx/dt and dv/dt using the Hamiltonian Neural Network.
    """
    q, p = x, -x  # Define momentum as -x
    H = model(q, p)
    
    dH_dq, dH_dp = torch.autograd.grad(H.sum(), (q, p), create_graph=True)
    
    dq_dt = dH_dp
    dp_dt = -dH_dq
    
    return dq_dt, dp_dt

def train_hnn(model, dataloader, epochs=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, v, target_x, target_v in dataloader:
            x.requires_grad_(True)
            v.requires_grad_(True)
            
            dq_dt, dp_dt = compute_time_derivatives(model, x)
            
            loss = loss_fn(dq_dt, target_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(dataloader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
    
    return losses

# Initialize model and train
DIMENSION = 1  # Adjust based on dataset
hnn = HamiltonianNN(DIMENSION)
dataloader = DataLoader(trajectories, batch_size=50, shuffle=True)
losses = train_hnn(hnn, dataloader)

# Plot results
fig5,ax5 = plt.subplots()
ax5.set_xlabel("p")
ax5.set_ylabel("H")
p_range = torch.linspace(-20,20,1000, dtype=torch.float32).reshape(-1,1)
zeros_column = torch.zeros_like(p_range, dtype=torch.float32).reshape(-1,1)

ax5.plot(p_range, hnn(zeros_column, p_range).detach(), label="learned")
ax5.set_title("Hamiltonian H(0,p)")
ax5.plot(p_range, 1/2 * p_range**2, label="analytic")
ax5.legend()


fig6,ax6 = plt.subplots()
ax6.set_xlabel("p")
ax6.set_ylabel("H")
p_range = torch.linspace(-20,20,1000, dtype=torch.float32).reshape(-1,1)
zeros_column = (torch.zeros_like(p_range, dtype=torch.float32) + 100).reshape(-1,1)

ax6.plot(p_range, hnn(zeros_column, p_range).detach(), label="learned")
ax6.set_title("Hamiltonian H(100,p)")
ax6.plot(p_range, 1/2 * p_range**2, label="analytic")
ax6.legend()

plt.show()