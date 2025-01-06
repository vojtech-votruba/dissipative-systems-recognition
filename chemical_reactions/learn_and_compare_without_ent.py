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
                self.velocity = torch.tensor(data[:,:,DIMENSION+1:-1], requires_grad=True).float()
                self.target_pos = torch.tensor(target[:,:,1:DIMENSION+1], requires_grad=True).float()
                self.target_vel = torch.tensor(target[:,:,DIMENSION+1:-1], requires_grad=True).float()

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

parser.add_argument("--epochs", default=100, type=int, help="number of epoches for the model to train")
parser.add_argument("--batch_size", default=50, type=int, help="batch size for training of the model")
parser.add_argument("--dt", default=0.01, type=float, help="size of the time step used in the simulation")
parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help="do you wish to train a new model?")
parser.add_argument('--plot', default=True, action=argparse.BooleanOptionalAction, help="option of plotting the loss function")
args = parser.parse_args()


torch.set_default_dtype(torch.float32)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DEVICE = "cpu"

torch.set_default_device(DEVICE)
print(f"Using {DEVICE} for tensor calculations")

if DEVICE == "cuda":
    print(f"CUDA version: {torch.version.cuda}")

# Defining the neural network
class EntropyNetwork(nn.Module):
    """
        For the entropy network we are using a fully input concave neural network achitecture,
        originally designed by myself - it's a simple alteration of FICNN - fully input convex neural nets,
        we just need to use concave, decreasing activation functions and negative weights instead of positive ones.
    """
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(DIMENSION, 50)

        self.prop_layer1 = nn.Linear(50, 50)
        self.lateral_layer1 = nn.Linear(DIMENSION, 50)

        self.prop_layer2 = nn.Linear(50, 50)
        self.lateral_layer2 = nn.Linear(DIMENSION, 50)

        self.prop_layer3 = nn.Linear(50, 50)
        self.lateral_layer3 = nn.Linear(DIMENSION, 50)

        self.prop_layer4 = nn.Linear(50, 50)
        self.lateral_layer4 = nn.Linear(DIMENSION, 50)

        self.prop_layer5 = nn.Linear(50, 50)
        self.lateral_layer5 = nn.Linear(DIMENSION, 50)

        self.output_layer = nn.Linear(50, 1)
        self.lateral_layer_out = nn.Linear(DIMENSION, 1)

    def forward(self, x0):
        x0 = x0.float()

        x = -nn.Softplus()(self.input_layer(x0))

        x = -nn.Softplus()(self.prop_layer1(x) + self.lateral_layer1(x0))
        self.prop_layer1.weight.data = -torch.abs(self.prop_layer1.weight.data)

        x = -nn.Softplus()(self.prop_layer2(x) + self.lateral_layer2(x0))
        self.prop_layer2.weight.data = -torch.abs(self.prop_layer2.weight.data)

        x = -nn.Softplus()(self.prop_layer3(x) + self.lateral_layer3(x0))
        self.prop_layer3.weight.data = -torch.abs(self.prop_layer3.weight.data)
                
        x = -nn.Softplus()(self.prop_layer4(x) + self.lateral_layer4(x0))
        self.prop_layer4.weight.data = -torch.abs(self.prop_layer4.weight.data)

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
        self.x_input_layer = nn.Linear(DIMENSION, 50)
        self.x_prop_layer1 = nn.Linear(50, 50)
        self.x_prop_layer2 = nn.Linear(50, 50)
        self.x_prop_layer3 = nn.Linear(50, 50)
        self.x_prop_layer4 = nn.Linear(50, 50)

        # The branch that goes directly between x and x_star
        self.x_lateral_layer_1 = nn.Linear(DIMENSION, 50)
        self.x_lateral_layer_2 = nn.Linear(50, 50)
        self.x_lateral_layer_3 = nn.Linear(50, 50)
        self.x_lateral_layer_4 = nn.Linear(50, 50)
        self.x_lateral_layer_5 = nn.Linear(50, 50)
        self.x_lateral_layer_out = nn.Linear(50, 1)

        # The branch that propagates x_star forward (We need to enforce convexity here)
        self.conjugate_prop_layer_1 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_2 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_3 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_4 = nn.Linear(50, 50, bias=False)
        self.conjugate_prop_layer_out= nn.Linear(50, 1, bias=False)

        self.conjugate_prop_layer_1_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_2_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_3_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_4_mid = nn.Linear(50, 50)
        self.conjugate_prop_layer_out_mid = nn.Linear(50, 50)

        # The branch which always starts at x0_star and ends at arbitrary x_star
        self.conjugate_lateral_layer_in = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_1 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_2 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_3 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_4 = nn.Linear(DIMENSION, 50, bias=False)
        self.conjugate_lateral_layer_out = nn.Linear(DIMENSION, 1, bias=False)

        self.conjugate_lateral_layer_in_mid = nn.Linear(DIMENSION, DIMENSION)
        self.conjugate_lateral_layer_1_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_2_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_3_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_4_mid = nn.Linear(50, DIMENSION)
        self.conjugate_lateral_layer_out_mid = nn.Linear(50, DIMENSION)

    def forward(self, input):
        input = input.float()

        x0 = input[:,:int(input.size(1)/2)]
        x0_star = input[:,int(input.size(1)/2):]

        x_star = nn.Softplus()(self.x_lateral_layer_1(x0) 
                               + self.conjugate_lateral_layer_in(torch.mul(x0_star, self.conjugate_lateral_layer_in_mid(x0))))
        x = nn.Softplus()(self.x_input_layer(x0))

        x_star = nn.Softplus()(self.x_lateral_layer_2(x) 
                               + self.conjugate_prop_layer_1(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_1_mid(x))))
                                + self.conjugate_lateral_layer_1(torch.mul(x0_star, self.conjugate_lateral_layer_1_mid(x))))
        x = nn.Softplus()(self.x_prop_layer1(x))
        self.conjugate_prop_layer_1.weight.data = torch.abs(self.conjugate_prop_layer_1.weight.data)

        x_star = nn.Softplus()(self.x_lateral_layer_3(x) 
                               + self.conjugate_prop_layer_2(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_2_mid(x))))
                                + self.conjugate_lateral_layer_2(torch.mul(x0_star, self.conjugate_lateral_layer_2_mid(x))))
        x = nn.Softplus()(self.x_prop_layer2(x))
        self.conjugate_prop_layer_2.weight.data = torch.abs(self.conjugate_prop_layer_2.weight.data)

        x_star = nn.Softplus()(self.x_lateral_layer_4(x) 
                               + self.conjugate_prop_layer_3(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_3_mid(x))))
                                + self.conjugate_lateral_layer_3(torch.mul(x0_star, self.conjugate_lateral_layer_3_mid(x))))
        x = nn.Softplus()(self.x_prop_layer3(x))
        self.conjugate_prop_layer_3.weight.data = torch.abs(self.conjugate_prop_layer_3.weight.data)

        x_star = nn.Softplus()(self.x_lateral_layer_5(x) 
                               + self.conjugate_prop_layer_4(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_4_mid(x))))
                                + self.conjugate_lateral_layer_4(torch.mul(x0_star, self.conjugate_lateral_layer_4_mid(x))))
        x = nn.Softplus()(self.x_prop_layer4(x))
        self.conjugate_prop_layer_4.weight.data = torch.abs(self.conjugate_prop_layer_4.weight.data)

        out = nn.Softplus()(self.x_lateral_layer_out(x) 
                            + self.conjugate_prop_layer_out(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_out_mid(x))))\
                                + self.conjugate_lateral_layer_out(torch.mul(x0_star, self.conjugate_lateral_layer_out_mid(x))))
        self.conjugate_prop_layer_out.weight.data = torch.abs(self.conjugate_prop_layer_out.weight.data)

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
        input = torch.stack((x,x_star), dim=1)

        Xi = self.Xi(input)
        x_dot = autograd.grad(Xi, x_star, grad_outputs=torch.ones_like(Xi), create_graph=True)[0]

        return [Xi,x_dot]
    
    def dissipation(self,x_star_tensor):
        x_star_tensor = x_star_tensor.float()

        ones_column = torch.ones_like(x_star_tensor, dtype=torch.float32)
        input = torch.stack((ones_column, x_star_tensor), dim=1)
        return self.Xi(input)

L = nn.MSELoss()

if args.train:
    start = time.time()
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(42)

    training_trajectories, test_trajectories = random_split(trajectories, [0.8, 0.2], generator=generator)
    dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size, shuffle=True, generator=generator)

    model = GradientDynamics().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training
    losses = []
    for i in range(args.epochs):
        for j, (traj_batch_pos, _, traj_batch_pos_target, traj_batch_vel_target) in enumerate(dataloader):
            optimizer.zero_grad()
            traj_batch_pos = traj_batch_pos.to(DEVICE)
            traj_batch_pos_target = traj_batch_pos_target.to(DEVICE)
            traj_batch_vel_target = traj_batch_vel_target.to(DEVICE)
            
            # Runge Kutta 4th order
            k1i = model(traj_batch_pos)[1]
            k2i = model(traj_batch_pos + k1i*args.dt/2)[1]
            k3i = model(traj_batch_pos + k2i*args.dt/2)[1]
            k4i = model(traj_batch_pos + k3i*args.dt)[1]
            predicted_velocity_rk = 1/6 * (k1i + 2*k2i + 2*k3i + k4i)

            predicted_velocity = model(traj_batch_pos)[1]

            loss = L(predicted_velocity, traj_batch_vel_target)
            
            """if torch.norm(predicted_velocity_rk - predicted_velocity) < 0.8:
                predicted_velocity = predicted_velocity_rk
                loss += L(predicted_velocity * args.dt + traj_batch_pos, traj_batch_pos_target)"""

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch no. {i}/{args.epochs} done!", end='\r')
    
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
    test_target = trajectories.target_pos[test_trajectories.indices].to(DEVICE)

else:
    test_data = trajectories.position.to(DEVICE)
    test_target = trajectories.target_pos.to(DEVICE)

MSE_test_set = L(model(test_data)[1]*args.dt+test_data, test_target).cpu()
print(f"MSE on the test set is: {MSE_test_set}")

if args.plot:
    plt.style.use('ggplot')
    # Plotting the MSE decline
    if args.train:
        fig1,ax1 = plt.subplots()
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("MSE")
        ax1.set_title("training loss decline on the training data")
        ax1.plot(range(len(losses)), losses)

    if DIMENSION == 1:
        # Sampling random trajectory and plotting it along with predicted trajectory
        fig2,ax2 = plt.subplots()
        sample = test_data[np.random.randint(0,len(test_data)-1)].cpu().detach().numpy()
        time = [args.dt*i for i in range(len(sample))]
    
        ax2.set_xlabel("t")
        ax2.set_ylabel("x")
        ax2.plot(time, sample, label="original data")

        prediction = [sample[0]]
        for i in range(len(sample)):
            prediction.append(float(model(torch.tensor([prediction[i]], 
                                                       requires_grad=True).unsqueeze(0))[1])*args.dt+prediction[i])

        prediction = np.array(prediction)
        ax2.set_title(f"MSE of the test set: {MSE_test_set}")
        ax2.plot(time[:-2], prediction[:-3] , label="prediction")
        ax2.legend()

        # Plotting dissipation potential
        fig3,ax3 = plt.subplots()
        ax3.set_xlabel("x*")
        ax3.set_ylabel("Ξ")
        x_star_range = torch.linspace(-50,50,200)
        ax3.plot(x_star_range, model.dissipation(x_star_range).cpu().detach())
        ax3.set_title("Dissipation potential Ξ = Ξ(x=0, x*)")

        # Plotting entropy
        fig4,ax4 = plt.subplots()
        ax4.set_xlabel("x")
        ax4.set_ylabel("S")
        x = torch.linspace(-50,50,500, dtype=torch.float32)
        x = x.view(-1, DIMENSION)
        ax4.plot(x, model.S(x).cpu().detach())
        ax4.set_title("Entropy S = S(x)")

    if DIMENSION == 2:
        # Sampling random trajectory and plotting it along with predicted trajectory
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection="3d")
        sample = test_data[np.random.randint(0,len(test_data)-1)].cpu().detach().numpy()
        time = [args.dt*i for i in range(len(sample))]
    
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        ax2.set_zlabel("t")

        ax2.plot(sample[:,0], sample[:,1], time, label="original data")

        prediction = [sample[0]]
        for i in range(len(sample)):
            prediction.append(np.array(model(torch.tensor([prediction[i]], 
                                                          requires_grad=True))[1].cpu().detach())[0]*args.dt+prediction[i])

        ax2.set_title(f"MSE of the test set: {MSE_test_set}")
        prediction = np.array(prediction)

        ax2.plot(prediction[:-3,0], prediction[:-3,1], time[:-2], label="prediction")
        ax2.legend()

        # Plotting dissipation potential
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(projection="3d")
        ax3.set_xlabel("x1*")
        ax3.set_ylabel("x2*")
        ax3.set_zlabel("log(Ξ)")

        x1_star = torch.linspace(-1000,1000,1000, dtype=torch.float32)
        x2_star = torch.linspace(-1000,1000,1000, dtype=torch.float32)

        X1_star, X2_star = torch.meshgrid(x1_star, x2_star, indexing="ij")
        X1_star_flat = X1_star.flatten()
        X2_star_flat = X2_star.flatten()
        points = torch.stack([X1_star_flat, X2_star_flat], dim=1)

        Xi_flat = model.dissipation(points)
        Xi = Xi_flat.reshape(X1_star.shape)

        X1_star_np = X1_star.cpu().numpy()
        X2_star_np = X2_star.cpu().numpy()
        Xi_np = np.log(Xi.cpu().detach().numpy())

        ax3.plot_surface(X1_star_np, X2_star_np, Xi_np, label="learned")
        ax3.set_title("Dissipation potential log(Ξ(1, x*))")

        # Plotting entropy
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(projection="3d")
        ax4.set_xlabel("x1")
        ax4.set_ylabel("x2")
        ax4.set_zlabel("S")

        x1 = torch.linspace(0,2000,1000, dtype=torch.float32)
        x2 = torch.linspace(0,2000,1000, dtype=torch.float32)

        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        X1_flat = X1.flatten()
        X2_flat = X2.flatten()
        points = torch.stack([X1_flat, X2_flat], dim=1)

        S_flat = model.S(points)
        S = S_flat.reshape(X1.shape)

        X1_np = X1.cpu().numpy()
        X2_np = X2.cpu().numpy()
        S_np = S.cpu().detach().numpy()

        ax4.plot_surface(X1_np, X2_np, S_np)
        ax4.set_title("Entropy S = S(x1, x2)")
        
    plt.show()
    