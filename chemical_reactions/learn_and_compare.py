# -*- coding: utf-8 -*-

import os
import time
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

# Adding some arguments
parser = argparse.ArgumentParser(prog="learn_and_test.py",
                                 description="A pytorch code for learning and testing phase space\
                                 trajectory prediciton.")

parser.add_argument("--epochs", default=200, type=int, help="number of epoches for the model to train")
parser.add_argument("--batch_size", default=50, type=int, help="batch size for training of the model")
parser.add_argument("--dt", default=0.01, type=float, help="size of the time step used in the simulation")
parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help="do you wish to train a new model?")
parser.add_argument('--log', default=False, action=argparse.BooleanOptionalAction, help="plot the log of dissipation potential")
args = parser.parse_args()

plt.style.use('ggplot')

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

                """self.data = torch.tensor(data[:,:,1:DIMENSION+1].reshape(
                    (data[:,:,1:DIMENSION+1].shape[0] * data[:,:,1:DIMENSION+1].shape[1], data[:,:,1:DIMENSION+1].shape[2])), requires_grad=True)
                
                self.target = torch.tensor(target[:,:,1:DIMENSION+1].reshape(
                    (target[:,:,1:DIMENSION+1].shape[0] * target[:,:,1:DIMENSION+1].shape[1], target[:,:,1:DIMENSION+1].shape[2])), requires_grad=True)"""
                
                self.n_samples = self.position.shape[0]

        def __getitem__(self, index):
            return self.position[index], self.velocity[index], self.target_pos[index], self.target_vel[index]

        def __len__(self):
            return self.n_samples

trajectories = TrajectoryDataset()

"""
    Plotting one of the trajectories to check if it has loaded correctely,
    and if it's numerically integrable by Euler's method with the chosen timestep

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

rand = np.random.randint(0,trajectories.__len__()-1)
random_trajectory = trajectories.position[rand].detach().numpy()
random_velocities = trajectories.velocity[rand].detach().numpy()

prediction = [random_trajectory[0]]
for v in random_velocities:
    prediction.append(prediction[-1] + v*args.dt)

times = [n*args.dt for n in range(len(random_trajectory))]

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("t")
ax.plot(random_trajectory[:,0], random_trajectory[:,1], times, label="original data")
ax.plot(random_trajectory[:,0], random_trajectory[:,1], times, label="numerical integration data")

plt.show()
"""

torch.set_default_dtype(torch.float32)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# My GPU doesn't have enough RAM to handle these computations
DEVICE = "cpu"

torch.set_default_device(DEVICE)
print(f"Using {DEVICE} for tensor calculations")

if DEVICE == "cuda":
    print(f"CUDA version: {torch.version.cuda}")

def enforce_pos(M):
    """
        This doesn't have anything to do with positive reinforcement,
        we just need a convinient, differentiable and continous way
        to enforce that all of the elements of a tensor are going to be positive
    """
    eps = torch.tensor(5.0, requires_grad=True)
    M[M >= 0] += torch.exp(-eps)
    M[M < 0] = torch.exp(M[M < 0] - eps)

    return M

"""
    Testing if the enforce_pos func. works

A = torch.tensor([[100,100000,1000000000],
                  [-100,-100000,-1000000000]])
print(enforce_pos(A.float()))
"""

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

class EntropyNetwork(nn.Module):
    """
        For the entropy network we are using a fully input concave neural network achitecture,
        it's a simple alteration of FICNN - fully input convex neural nets,
        we just need to use concave, decreasing activation functions and negative weights instead of positive ones.
    """

    def __init__(self):
        super().__init__()

        """
        It seems we can't really use them because of the needed convexity

        Dropout layers
        self.dropout_layer1 = nn.Dropout(p=0.1)
        self.dropout_layer2 = nn.Dropout(p=0.1)
        self.dropout_layer3 = nn.Dropout(p=0.1)
        self.dropout_layer4 = nn.Dropout(p=0.1)
        """

        self.input_layer = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.input_layer.weight)

        self.prop_layer1 = nn.Linear(50, 50)
        self.lateral_layer1 = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.prop_layer1.weight)
        nn.init.xavier_normal_(self.lateral_layer1.weight)

        self.prop_layer2 = nn.Linear(50, 50)
        self.lateral_layer2 = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.prop_layer2.weight)
        nn.init.xavier_normal_(self.lateral_layer2.weight)

        self.prop_layer3 = nn.Linear(50, 50)
        self.lateral_layer3 = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.prop_layer3.weight)
        nn.init.xavier_normal_(self.lateral_layer3.weight)

        self.prop_layer4 = nn.Linear(50, 50)
        self.lateral_layer4 = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.prop_layer4.weight)
        nn.init.xavier_normal_(self.lateral_layer4.weight)

        self.prop_layer5 = nn.Linear(50, 50)
        self.lateral_layer5 = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.prop_layer5.weight)
        nn.init.xavier_normal_(self.lateral_layer5.weight)

        self.prop_layer6 = nn.Linear(50, 50)
        self.lateral_layer6 = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.prop_layer6.weight)
        nn.init.xavier_normal_(self.lateral_layer6.weight)

        self.prop_layer7 = nn.Linear(50, 50)
        self.lateral_layer7 = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.prop_layer7.weight)
        nn.init.xavier_normal_(self.lateral_layer7.weight)

        self.prop_layer8 = nn.Linear(50, 50)
        self.lateral_layer8 = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.prop_layer8.weight)
        nn.init.xavier_normal_(self.lateral_layer8.weight)

        self.prop_layer9 = nn.Linear(50, 50)
        self.lateral_layer9 = nn.Linear(DIMENSION, 50)
        nn.init.xavier_normal_(self.prop_layer9.weight)
        nn.init.xavier_normal_(self.lateral_layer9.weight)

        self.output_layer = nn.Linear(50, 1)
        self.lateral_layer_out = nn.Linear(DIMENSION, 1)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.xavier_normal_(self.lateral_layer_out.weight)

    def forward(self, x0):

        x0 = x0.float()

        x = -nn.Softplus()(self.input_layer(x0))

        x = -nn.Softplus()(self.prop_layer1(x) + self.lateral_layer1(x0))
        self.prop_layer1.weight.data = -enforce_pos(self.prop_layer1.weight.data)
        #x = self.dropout_layer1(x)

        x = -nn.Softplus()(self.prop_layer2(x) + self.lateral_layer2(x0))
        self.prop_layer2.weight.data = -enforce_pos(self.prop_layer2.weight.data)
        #x = self.dropout_layer2(x)

        x = -nn.Softplus()(self.prop_layer3(x) + self.lateral_layer3(x0))
        self.prop_layer3.weight.data = -enforce_pos(self.prop_layer3.weight.data)
        #x = self.dropout_layer3(x)
                
        x = -nn.Softplus()(self.prop_layer4(x) + self.lateral_layer4(x0))
        self.prop_layer4.weight.data = -enforce_pos(self.prop_layer4.weight.data)
        #x = self.dropout_layer4(x)

        x = -nn.Softplus()(self.prop_layer5(x) + self.lateral_layer5(x0))
        self.prop_layer5.weight.data = -enforce_pos(self.prop_layer5.weight.data)

        x = -nn.Softplus()(self.prop_layer6(x) + self.lateral_layer6(x0))
        self.prop_layer6.weight.data = -enforce_pos(self.prop_layer6.weight.data)

        x = -nn.Softplus()(self.prop_layer7(x) + self.lateral_layer7(x0))
        self.prop_layer7.weight.data = -enforce_pos(self.prop_layer7.weight.data)

        x = -nn.Softplus()(self.prop_layer8(x) + self.lateral_layer8(x0))
        self.prop_layer8.weight.data = -enforce_pos(self.prop_layer8.weight.data)

        x = -nn.Softplus()(self.prop_layer9(x) + self.lateral_layer9(x0))
        self.prop_layer9.weight.data = -enforce_pos(self.prop_layer9.weight.data)

        S = -nn.Softplus()(self.output_layer(x) + self.lateral_layer_out(x0))
        self.output_layer.weight.data = -enforce_pos(self.output_layer.weight.data)
        
        return S

# Really the main part of the network
class DissipationNetwork(nn.Module):
    """
        For this network we are using a more complex architecture to ensure 
        only a partial convexity of the output with respect to some inputs.
        Specifically: PICNN, source: https://arxiv.org/pdf/1609.07152
    """

    def __init__(self):
        super().__init__()
        
        """ 
        It seems we can't really use them because of the needed convexity

        Dropout layers
        self.dropout_layer1 = nn.Dropout(p=0.1)
        self.dropout_layer2 = nn.Dropout(p=0.1)
        self.dropout_layer3 = nn.Dropout(p=0.1)
        self.dropout_layer4 = nn.Dropout(p=0.1)
        self.dropout_layer5 = nn.Dropout(p=0.1)
        """

        # The branch that propagates x directly forward
        self.x_input_layer = nn.Linear(DIMENSION, 50)
        self.x_prop_layer1 = nn.Linear(50, 50)
        self.x_prop_layer2 = nn.Linear(50, 50)
        self.x_prop_layer3 = nn.Linear(50, 50)
        self.x_prop_layer4 = nn.Linear(50, 50)
        self.x_prop_layer5 = nn.Linear(50, 50)
        self.x_prop_layer6 = nn.Linear(50, 50)
        self.x_prop_layer7 = nn.Linear(50, 50)
        self.x_prop_layer8 = nn.Linear(50, 50)
        self.x_prop_layer9 = nn.Linear(50, 50)
        self.x_prop_layer10 = nn.Linear(50, 50)

        nn.init.xavier_normal_(self.x_input_layer.weight)
        nn.init.xavier_normal_(self.x_prop_layer1.weight)
        nn.init.xavier_normal_(self.x_prop_layer2.weight)
        nn.init.xavier_normal_(self.x_prop_layer3.weight)
        nn.init.xavier_normal_(self.x_prop_layer4.weight)
        nn.init.xavier_normal_(self.x_prop_layer5.weight)
        nn.init.xavier_normal_(self.x_prop_layer6.weight)
        nn.init.xavier_normal_(self.x_prop_layer7.weight)
        nn.init.xavier_normal_(self.x_prop_layer8.weight)
        nn.init.xavier_normal_(self.x_prop_layer9.weight)
        nn.init.xavier_normal_(self.x_prop_layer10.weight)

        # The branch that goes directly between x and x_star
        self.x_lateral_layer_1 = nn.Linear(DIMENSION, 50)
        self.x_lateral_layer_2 = nn.Linear(50, 50)
        self.x_lateral_layer_3 = nn.Linear(50, 50)
        self.x_lateral_layer_4 = nn.Linear(50, 50)
        self.x_lateral_layer_5 = nn.Linear(50, 50)
        self.x_lateral_layer_6 = nn.Linear(50, 50)
        self.x_lateral_layer_7 = nn.Linear(50, 50)
        self.x_lateral_layer_8 = nn.Linear(50, 50)
        self.x_lateral_layer_9 = nn.Linear(50, 50)
        self.x_lateral_layer_10 = nn.Linear(50, 50)
        self.x_lateral_layer_11 = nn.Linear(50, 50)
        self.x_lateral_layer_out = nn.Linear(50, 1)

        nn.init.xavier_normal_(self.x_lateral_layer_1.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_2.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_3.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_4.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_5.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_6.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_7.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_8.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_9.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_10.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_11.weight)
        nn.init.xavier_normal_(self.x_lateral_layer_out.weight)

        # The branch that propagates x_star forward (We need to enforce convexity here)
        self.conjugate_prop_layer_1 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_2 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_3 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_4 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_5 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_6 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_7 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_8 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_9 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_10 = nn.Linear(50, 50, bias=False)             
        self.conjugate_prop_layer_out= nn.Linear(50, 1, bias=False)

        nn.init.xavier_normal_(self.conjugate_prop_layer_1.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_2.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_3.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_4.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_5.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_6.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_7.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_8.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_9.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_10.weight)      
        nn.init.xavier_normal_(self.conjugate_prop_layer_out.weight)

        self.conjugate_prop_layer_1_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_2_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_3_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_4_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_5_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_6_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_7_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_8_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_9_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_10_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_out_mid = nn.Linear(50, 50)

        nn.init.xavier_normal_(self.conjugate_prop_layer_1_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_2_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_3_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_4_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_5_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_6_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_7_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_8_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_9_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_10_mid.weight)
        nn.init.xavier_normal_(self.conjugate_prop_layer_out_mid.weight)

        # The branch which always starts at x0_star and ends at arbitrary x_star
        self.conjugate_lateral_layer_in = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_1 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_2 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_3 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_4 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_5 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_6 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_7 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_8 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_9 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_10 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_out = nn.Linear(DIMENSION, 1, bias=False)

        nn.init.xavier_normal_(self.conjugate_lateral_layer_in.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_1.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_2.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_3.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_4.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_5.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_6.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_7.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_8.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_9.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_10.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_out.weight)


        self.conjugate_lateral_layer_in_mid = nn.Linear(DIMENSION, DIMENSION)
        self.conjugate_lateral_layer_1_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_2_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_3_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_4_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_5_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_6_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_7_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_8_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_9_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_10_mid = nn.Linear(50, DIMENSION)       
        self.conjugate_lateral_layer_out_mid = nn.Linear(50, DIMENSION)

        nn.init.xavier_normal_(self.conjugate_lateral_layer_in_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_1_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_2_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_3_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_4_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_5_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_6_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_7_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_8_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_9_mid.weight)
        nn.init.xavier_normal_(self.conjugate_lateral_layer_10_mid.weight)      
        nn.init.xavier_normal_(self.conjugate_lateral_layer_out_mid.weight)

    def forward(self, input):
        input = input.float()

        x0, x0_star = torch.split(input, 2, dim=2) 

        x_star = nn.Softplus()(self.x_lateral_layer_1(x0) 
                               + self.conjugate_lateral_layer_in(torch.mul(x0_star, self.conjugate_lateral_layer_in_mid(x0))))
        x = nn.Softplus()(self.x_input_layer(x0))
        
        x_star = nn.Softplus()(self.x_lateral_layer_2(x) 
                               + self.conjugate_prop_layer_1(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_1_mid(x))))
                                + self.conjugate_lateral_layer_1(torch.mul(x0_star, self.conjugate_lateral_layer_1_mid(x))))
        x = nn.Softplus()(self.x_prop_layer1(x))
        self.conjugate_prop_layer_1.weight.data = enforce_pos(self.conjugate_prop_layer_1.weight.data)
        #x = self.dropout_layer1(x)
        #x_star = self.dropout_layer1(x)

        x_star = nn.Softplus()(self.x_lateral_layer_3(x) 
                               + self.conjugate_prop_layer_2(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_2_mid(x))))
                                + self.conjugate_lateral_layer_2(torch.mul(x0_star, self.conjugate_lateral_layer_2_mid(x))))
        x = nn.Softplus()(self.x_prop_layer2(x))
        self.conjugate_prop_layer_2.weight.data = enforce_pos(self.conjugate_prop_layer_2.weight.data)
        #x = self.dropout_layer2(x)
        #x_star = self.dropout_layer2(x)

        x_star = nn.Softplus()(self.x_lateral_layer_4(x) 
                               + self.conjugate_prop_layer_3(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_3_mid(x))))
                                + self.conjugate_lateral_layer_3(torch.mul(x0_star, self.conjugate_lateral_layer_3_mid(x))))
        x = nn.Softplus()(self.x_prop_layer3(x))
        self.conjugate_prop_layer_3.weight.data = enforce_pos(self.conjugate_prop_layer_3.weight.data)
        #x = self.dropout_layer3(x)
        #x_star = self.dropout_layer3(x)

        x_star = nn.Softplus()(self.x_lateral_layer_5(x) 
                               + self.conjugate_prop_layer_4(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_4_mid(x))))
                                + self.conjugate_lateral_layer_4(torch.mul(x0_star, self.conjugate_lateral_layer_4_mid(x))))
        x = nn.Softplus()(self.x_prop_layer4(x))
        self.conjugate_prop_layer_4.weight.data = enforce_pos(self.conjugate_prop_layer_4.weight.data)
        #x = self.dropout_layer4(x)
        #x_star = self.dropout_layer4(x)

        x_star = nn.Softplus()(self.x_lateral_layer_6(x) 
                               + self.conjugate_prop_layer_5(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_5_mid(x))))
                                + self.conjugate_lateral_layer_5(torch.mul(x0_star, self.conjugate_lateral_layer_5_mid(x))))
        x = nn.Softplus()(self.x_prop_layer5(x))
        self.conjugate_prop_layer_5.weight.data = enforce_pos(self.conjugate_prop_layer_5.weight.data)

        x_star = nn.Softplus()(self.x_lateral_layer_7(x) 
                               + self.conjugate_prop_layer_6(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_6_mid(x))))
                                + self.conjugate_lateral_layer_6(torch.mul(x0_star, self.conjugate_lateral_layer_6_mid(x))))
        x = nn.Softplus()(self.x_prop_layer6(x))
        self.conjugate_prop_layer_6.weight.data = enforce_pos(self.conjugate_prop_layer_6.weight.data)
        
        x_star = nn.Softplus()(self.x_lateral_layer_8(x) 
                               + self.conjugate_prop_layer_7(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_7_mid(x))))
                                + self.conjugate_lateral_layer_7(torch.mul(x0_star, self.conjugate_lateral_layer_7_mid(x))))
        x = nn.Softplus()(self.x_prop_layer7(x))
        self.conjugate_prop_layer_7.weight.data = enforce_pos(self.conjugate_prop_layer_7.weight.data)

        x_star = nn.Softplus()(self.x_lateral_layer_9(x) 
                               + self.conjugate_prop_layer_8(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_8_mid(x))))
                                + self.conjugate_lateral_layer_8(torch.mul(x0_star, self.conjugate_lateral_layer_8_mid(x))))
        x = nn.Softplus()(self.x_prop_layer8(x))
        self.conjugate_prop_layer_8.weight.data = enforce_pos(self.conjugate_prop_layer_8.weight.data)

        x_star = nn.Softplus()(self.x_lateral_layer_10(x) 
                               + self.conjugate_prop_layer_9(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_9_mid(x))))
                                + self.conjugate_lateral_layer_9(torch.mul(x0_star, self.conjugate_lateral_layer_9_mid(x))))
        x = nn.Softplus()(self.x_prop_layer9(x))
        self.conjugate_prop_layer_9.weight.data = enforce_pos(self.conjugate_prop_layer_9.weight.data)

        x_star = nn.Softplus()(self.x_lateral_layer_11(x) 
                               + self.conjugate_prop_layer_10(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_10_mid(x))))
                                + self.conjugate_lateral_layer_10(torch.mul(x0_star, self.conjugate_lateral_layer_10_mid(x))))
        x = nn.Softplus()(self.x_prop_layer10(x))
        self.conjugate_prop_layer_10.weight.data = enforce_pos(self.conjugate_prop_layer_10.weight.data)

        out = nn.Softplus()(self.x_lateral_layer_out(x) 
                            + self.conjugate_prop_layer_out(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_out_mid(x))))\
                                + self.conjugate_lateral_layer_out(torch.mul(x0_star, self.conjugate_lateral_layer_out_mid(x))))
        self.conjugate_prop_layer_out.weight.data = enforce_pos(self.conjugate_prop_layer_out.weight.data)

        return out

class GradientDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = EntropyNetwork()
        self.Xi = DissipationNetwork()

    def forward(self, x):
        x = x.float()

        S = self.S(x)
        x_star = autograd.grad(S, x, grad_outputs=torch.ones_like(S), create_graph=True)[0]
        input = torch.cat((x,x_star), dim=2)

        Xi = self.Xi(input)
        x_dot = autograd.grad(Xi, x_star, grad_outputs=torch.ones_like(Xi), create_graph=True)[0]

        return [Xi,x_dot]

L = nn.MSELoss()

if args.train:
    start = time.time()
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(42)

    training_trajectories, test_trajectories = random_split(trajectories, [0.8, 0.2], generator=generator)
    dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size, shuffle=True, generator=generator)

    model = GradientDynamics().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)

    """
        Testing if the model intialized correctely

    for name, param in model.named_parameters():
        if "weight" in name:  # Focus on weight matrices
            frobenius_norm = param.data.norm('fro')
            print(f"Frobenius norm of {name}: {frobenius_norm.item()}")
    """

    # Training
    losses = []
    log_losses = []

    for i in range(args.epochs):

        for j, (traj_batch_pos, traj_batch_velocity, _, _) in enumerate(dataloader):
            optimizer.zero_grad()

            traj_batch_pos = traj_batch_pos.to(DEVICE)
            traj_batch_velocity = traj_batch_velocity.to(DEVICE)
            
            # Runge Kutta 4th order
            #predicted_velocity_rk = rk4(model, traj_batch_pos, args.dt)
            Xi, predicted_velocity = model(traj_batch_pos)
            dXi = autograd.grad(Xi, traj_batch_pos, grad_outputs=torch.ones_like(Xi), create_graph=True)[0][:,:-1]
            
            S = model.S(traj_batch_pos[0])
            x_star = autograd.grad(S, traj_batch_pos, grad_outputs=torch.ones_like(S), create_graph=True)[0]
            x_star_dot = (x_star[:,1:] - x_star[:,:-1]) / args.dt

            """
                The loss comprises of two parts: the standard MSE for predicted vs. real velocity
                and the symplectic loss which is imposing a symplectic structure
                upon the dissipation potential in accordance with the geometrical origin of grad. dynamics
            """
            loss = L(predicted_velocity, traj_batch_velocity) + L(x_star_dot, -dXi) #+ L(predicted_velocity_rk, predicted_velocity)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()
            losses.append(loss.item())
            log_losses.append(np.log(loss.item()))

        print(f"Epoch no. {i}/{args.epochs} done! The loss for the last batch in this epoch was: {loss}")

    stop = time.time()
    print(f"\nTraining took {stop - start} sec")

    if os.path.exists("models"):
        torch.save(model.state_dict(), "models/model.pth")
    else:
        os.mkdir("models")
        torch.save(model.state_dict(), "models/model.pth")
else:
    model = GradientDynamics().to(DEVICE)
    model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    model.eval()

if args.train:
    test_data = trajectories.position[test_trajectories.indices].to(DEVICE)
    test_target = trajectories.target_vel[test_trajectories.indices].to(DEVICE)

else:
    test_data = trajectories.position.to(DEVICE)
    test_target = trajectories.target_vel.to(DEVICE)

MSE_test_set = L(model(test_data)[1], test_target)
print(f"MSE on the test set is: {MSE_test_set}")

if args.train:
    fig2,ax2 = plt.subplots()
    ax2.plot(log_losses)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("log(MSE)")
    ax2.set_title("log of training loss)")

if DIMENSION == 1:
    # Sampling random trajectory and plotting it along with predicted trajectory
    fig3,ax3 = plt.subplots()
    sample = test_data[np.random.randint(0,len(test_data)-1)].cpu().detach().numpy()
    time = [args.dt*i for i in range(len(sample))]

    ax3.set_xlabel("t")
    ax3.set_ylabel("x")
    ax3.plot(time, sample, label="original data")

    prediction = [sample[0]]
    for i in range(len(sample)):
        prediction.append(float(model(torch.tensor([prediction[i]], 
                                                    requires_grad=True).unsqueeze(0))[1])*args.dt+prediction[i])

    prediction = np.array(prediction)
    ax3.set_title(f"MSE of the test set: {MSE_test_set}")
    ax3.plot(time[:-2], prediction[:-3] , label="prediction")
    ax3.legend()

    # Plotting dissipation potential
    fig4,ax4 = plt.subplots()
    ax4.set_xlabel("x*")
    ax4.set_ylabel("Ξ")
    x_star_range = torch.linspace(-5,5,200)
    ax4.plot(x_star_range, model.dissipation(x_star_range).cpu().detach())
    ax4.set_title("Dissipation potential Ξ = Ξ(x=0, x*)")

    # Plotting entropy
    fig5,ax5 = plt.subplots()
    ax5.set_xlabel("x")
    ax5.set_ylabel("S")
    x = torch.linspace(-50,50,500, dtype=torch.float32)
    x = x.view(-1, DIMENSION)
    ax5.plot(x, model.S(x).cpu().detach())
    ax5.set_title("Entropy S = S(x)")

if DIMENSION == 2:
    # Sampling random trajectory and plotting it along with predicted trajectory
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection="3d")
    sample = test_data[np.random.randint(0,len(test_data)-1)].cpu().detach().numpy()
    time = [args.dt*i for i in range(len(sample))]

    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_zlabel("t")

    ax3.plot(sample[:,0], sample[:,1], time, label="original data")
    velocities = model(torch.tensor([sample], requires_grad=True))[1]

    prediction = [sample[0]]

    for i in range(len(sample)):
        prediction.append(prediction[i] + args.dt * velocities[0][i].detach().numpy())

    ax3.set_title(f"MSE of the test set: {MSE_test_set}")
    prediction = np.array(prediction)

    ax3.plot(prediction[:-3,0], prediction[:-3,1], time[:-2], label="prediction")
    ax3.legend()

    # Plotting dissipation potential
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(projection="3d")
    ax4.set_xlabel("x1*")
    ax4.set_ylabel("x2*")

    x1_star = torch.linspace(-500,500,1000, dtype=torch.float32)
    x2_star = torch.linspace(-500,500,1000, dtype=torch.float32)

    X1_star, X2_star = torch.meshgrid(x1_star, x2_star, indexing="ij")
    X1_star_flat = X1_star.flatten()
    X2_star_flat = X2_star.flatten()
    points = torch.stack([X1_star_flat, X2_star_flat], dim=1)

    ones_column = torch.ones_like(points, dtype=torch.float32)
    input = torch.cat((ones_column, points), dim=1).unsqueeze(0)

    Xi_flat = model.Xi(input)
    Xi = Xi_flat.reshape(X1_star.shape)

    X1_star_np = X1_star.cpu().numpy()
    X2_star_np = X2_star.cpu().numpy()
    Xi_np = Xi.cpu().detach().numpy()
    ax4.set_title("Dissipation potential Ξ(1, x*)")
    Xi_theor = np.cosh((X2_star_np - X1_star_np)/2)

    if args.log:
        ax4.set_zlabel("log(Ξ)")
        Xi_np = np.log(Xi_np)
        Xi_theor = np.log(Xi_theor)
        ax4.set_title("Dissipation potential log(Ξ(1, x*))")

    ax4.plot_surface(X1_star_np, X2_star_np, Xi_np, label="learned")
    ax4.plot_surface(X1_star_np, X2_star_np, Xi_theor , label="theoretical")
    ax4.legend() 

    # Plotting entropy
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(projection="3d")
    ax5.set_xlabel("x1")
    ax5.set_ylabel("x2")
    ax5.set_zlabel("S")

    x1 = torch.linspace(0.05,1000,1000, dtype=torch.float32)
    x2 = torch.linspace(0.05,1000,1000, dtype=torch.float32)

    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    X1_flat = X1.flatten()
    X2_flat = X2.flatten()
    points = torch.stack([X1_flat, X2_flat], dim=1)

    S_flat = model.S(points)
    S = S_flat.reshape(X1.shape)

    X1_np = X1.cpu().numpy()
    X2_np = X2.cpu().numpy()
    S_np = S.cpu().detach().numpy()

    S_theor = -X1_np*(np.log(X1_np) - 1) - X2_np*(np.log(X2_np) - 1)

    ax5.plot_surface(X1_np, X2_np, S_np, label="learned")
    ax5.plot_surface(X1_np, X2_np, S_theor, label="theoretical")    
    ax5.legend()

    ax5.set_title("Entropy S = S(x1, x2)")

plt.show()
