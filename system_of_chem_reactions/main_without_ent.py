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

# Adding some arguments
parser = argparse.ArgumentParser(prog="learn_and_test.py",
                                 description="A pytorch code for learning and testing state space\
                                 trajectory prediciton.")

parser.add_argument("--epochs", default=1000, type=int, help="number of epoches for the model to train")
parser.add_argument("--batch_size", default=128, type=int, help="batch size for training of the model")
parser.add_argument("--dt", default=0.002, type=float, help="size of the time step used in the simulation")
parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help="do you wish to train a new model?")
parser.add_argument('--plot', default=True, action=argparse.BooleanOptionalAction, help="option of plotting the loss function")
parser.add_argument("--log", default=True, type=int, help="using log loss for plotting and such")
parser.add_argument("--eps", default=5.0, type=float, help="small epsilon used for weights reparametrization")
parser.add_argument("--lbfgs", default=True, action=argparse.BooleanOptionalAction, help="use lbfgs for optimalization")

args = parser.parse_args()

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

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_device(DEVICE)
print(f"Using {DEVICE} for tensor calculations")

generator = torch.Generator(device=DEVICE)
generator.manual_seed(42)

class PositiveLinear(nn.Linear):
    """
        A positive layer that we use to enforce convexity
    """

    def forward(self, input):
        W = self.weight
        eps_tensor = torch.tensor(args.eps, device=W.device, dtype=W.dtype)

        positive_W = W + torch.exp(-eps_tensor)
        negative_W = torch.exp(W - eps_tensor) 
        reparam_W = torch.where(W >= 0, positive_W, negative_W) 

        return nn.functional.linear(input, reparam_W, self.bias)

def rk4(f, x, time_step):
    """
        Classical 4th order Runge Kutta implementation
    """
    k1i = f(x)
    k2i = f(x + k1i * time_step/2)
    k3i = f(x + k2i * time_step/2)
    k4i = f(x + k3i * time_step)

    return 1/6 * (k1i + 2*k2i + 2*k3i + k4i)

def conjugate(x):
    return -torch.log(x)

class DissipationNetwork(nn.Module):
    """
        For this network we are using a more complex architecture to ensure 
        only a partial convexity of the output with respect to some inputs.
        Specifically: PICNN, source: https://arxiv.org/pdf/1609.07152
    """
    def __init__(self):
        super().__init__()
        # The branch that propagates x directly forward
        self.x_input_layer = nn.Linear(DIMENSION, 6)
        self.x_prop_layer1 = nn.Linear(6, 6)

        # The branch that goes directly between x and x_star
        self.x_lateral_layer_1 = nn.Linear(DIMENSION, 6)
        self.x_lateral_layer_2 = nn.Linear(6, 6)
        self.x_lateral_layer_out = nn.Linear(6, 1)

        # The branch that propagates x_star forward (We need to enforce convexity here)
        self.conjugate_prop_layer_1 = PositiveLinear(6, 6, bias=False)
        self.conjugate_prop_layer_out= PositiveLinear(6, 1, bias=False)

        self.conjugate_prop_layer_1_mid = nn.Linear(6, 6)
        self.conjugate_prop_layer_out_mid = nn.Linear(6, 6)

        # The branch which always starts at x0_star and ends at arbitrary x_star
        self.conjugate_lateral_layer_in = nn.Linear(DIMENSION, 6, bias=False)
        self.conjugate_lateral_layer_1 = nn.Linear(DIMENSION, 6, bias=False)
        self.conjugate_lateral_layer_out = nn.Linear(DIMENSION, 1, bias=False)

        self.conjugate_lateral_layer_in_mid = nn.Linear(DIMENSION, DIMENSION)
        self.conjugate_lateral_layer_1_mid = nn.Linear(6, DIMENSION)
        self.conjugate_lateral_layer_out_mid = nn.Linear(6, DIMENSION)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state, state_conjugate):
        x0 = state
        x0_star = state_conjugate

        x_star = nn.Softplus()(self.x_lateral_layer_1(x0) 
                               + self.conjugate_lateral_layer_in(torch.mul(x0_star, self.conjugate_lateral_layer_in_mid(x0))))
        x = nn.Softplus()(self.x_input_layer(x0))

        x_star = nn.Softplus()(self.x_lateral_layer_2(x) 
                               + self.conjugate_prop_layer_1(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_1_mid(x))))
                                + self.conjugate_lateral_layer_1(torch.mul(x0_star, self.conjugate_lateral_layer_1_mid(x))))
        x = nn.Softplus()(self.x_prop_layer1(x))

        Xi_out = nn.Softplus()(self.x_lateral_layer_out(x) 
                            + self.conjugate_prop_layer_out(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_out_mid(x))))\
                                + self.conjugate_lateral_layer_out(torch.mul(x0_star, self.conjugate_lateral_layer_out_mid(x))))

        return Xi_out

class GradientDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.Xi = DissipationNetwork()

    def forward(self, x, x_star):
        x_star_zeros = torch.zeros_like(x, requires_grad=True)
        Xi_raw = self.Xi(x, x_star)
        Xi_at_zero = self.Xi(x, x_star_zeros)
        Xi = Xi_raw - Xi_at_zero - (x_star * autograd.grad(Xi_at_zero, x_star_zeros, grad_outputs=torch.ones_like(Xi_at_zero), create_graph=True)[0]).sum(dim=-1).unsqueeze(-1)

        return Xi
    
    def predict(self, x):
        x_star = conjugate(x)
        Xi = self.forward(x,x_star)
        x_dot = autograd.grad(Xi, x_star, grad_outputs=torch.ones_like(Xi), create_graph=True)[0]

        return x_dot

L = nn.MSELoss()

if args.train:
    training_trajectories, test_trajectories = random_split(trajectories, [0.8, 0.2], generator=generator)
    model = GradientDynamics().to(DEVICE)

    lbfgs_dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size, shuffle=True, generator=generator)
    adam_dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size // 2, shuffle=True, generator=generator)

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-1, max_iter=10, history_size=20, line_search_fn='strong_wolfe')

    # Training
    trajectory_losses = []
    velocity_losses = []

    for i in range(args.epochs):
        if i < args.epochs // 1.5 or not args.lbfgs:
            dataloader = adam_dataloader
            optimizer = adam_optimizer
        else:
            dataloader = lbfgs_dataloader
            optimizer = lbfgs_optimizer

        for j, (pos, veloc, targ_pos, targ_veloc) in enumerate(dataloader):
            pos = pos.to(DEVICE)
            veloc = veloc.to(DEVICE)
            targ_pos = targ_pos.to(DEVICE)
            targ_veloc = targ_veloc.to(DEVICE)

            if i < args.epochs // 1.5 or not args.lbfgs:
                optimizer.zero_grad()

                predicted_veloc = rk4(model.predict, pos, args.dt)
                trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos) / torch.std(targ_pos)
                velocity_loss = L(predicted_veloc, veloc) / torch.std(veloc)

                loss = velocity_loss
                loss.backward()
                optimizer.step()

            else:
                def closure():
                    optimizer.zero_grad()

                    predicted_veloc = rk4(model.predict, pos, args.dt)
                    trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos) / torch.std(targ_pos)
                    velocity_loss = L(predicted_veloc, veloc) / torch.std(veloc)

                    loss = velocity_loss
                    loss.backward()
                    return loss

                optimizer.step(closure)

            predicted_veloc = rk4(model.predict, pos, args.dt)
            trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos)
            velocity_loss = L(predicted_veloc, veloc)

            if args.log:
                trajectory_losses.append(np.log(trajectory_loss.item()))
                velocity_losses.append(np.log(velocity_loss.item()))
            else:
                trajectory_losses.append(trajectory_loss.item())
                velocity_losses.append(velocity_loss.item())

        print(f"Epoch no. {i}/{args.epochs} done! Traj. loss: {trajectory_loss}. Vel. loss: {velocity_loss}.")

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
    test_pos = trajectories.position[test_trajectories.indices].to(DEVICE)
    test_vel = trajectories.velocity[test_trajectories.indices].to(DEVICE)
    test_target_pos = trajectories.target_pos[test_trajectories.indices].to(DEVICE)
    test_target_vel = trajectories.target_vel[test_trajectories.indices].to(DEVICE)

else:
    test_pos = trajectories.position.to(DEVICE)
    test_vel = trajectories.velocity.to(DEVICE)
    test_target_pos = trajectories.target_pos.to(DEVICE)
    test_target_vel = trajectories.target_vel.to(DEVICE)

MSE_trajectory_loss = L(rk4(model.predict, test_pos, args.dt) + test_pos, test_target_pos)
MSE_velocity_loss = L(rk4(model.predict, test_pos, args.dt), test_vel)
print(f"Trajectory loss on the test set is {MSE_trajectory_loss}. Velocity loss on the test set is {MSE_velocity_loss}.")

if args.plot:
    plt.style.use("bmh")
    custom_cycler = plt.cycler(
        color=['#6c3b9c', '#a02c2c', '#2c7a2c', '#2c2c7a', '#287d7d', '#aa5500', '#555555'],
        linestyle=['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
    )
    plt.rc('axes', prop_cycle=custom_cycler)

    # Plotting the MSE decline
    if args.train:
        fig0,ax0 = plt.subplots()
        ax0.set_xlabel("Iterations")
        ax0.set_ylabel("MSE")
        if args.log:
            ax0.set_title("Training log loss decline on the training data")
        else:
            ax0.set_title("Training loss decline on the training data")

        ax0.plot(range(len(trajectory_losses)), trajectory_losses, label="trajectory loss")
        ax0.plot(range(len(velocity_losses)), velocity_losses, label="velocity loss")
        ax0.legend()

    stoichiometric_matrix = torch.tensor([
    [-1.0,+0.0],
    [-1.0,+1.0],
    [+1.0,-1.0],
    [+0.0,-1.0],
    [+0.0,+1.0],], dtype=torch.float32)

    if DIMENSION == 2:
        # Sampling random trajectory and plotting it along with predicted trajectory
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection="3d")
        sample = test_pos[np.random.randint(0,len(test_pos)-1)].cpu().detach().numpy()
        tensor_sample = torch.tensor([sample], requires_grad=True)
        time_set = [args.dt*i for i in range(len(sample))]

        ax1.set_xlabel("c1")
        ax1.set_ylabel("c2")
        ax1.set_zlabel("t")
        ax1.set_title(f"Sample trajectory")

        prediction = [sample[0]]
        print("calculating sample trajectory... it shouldn't take too long")
        for i in range(len(sample)):
            velocity = rk4(model.predict, torch.tensor(prediction[i], requires_grad=True), args.dt)
            prediction.append(prediction[i] + args.dt * velocity.cpu().detach().numpy())

        prediction = np.array(prediction)

        ax1.plot(sample[:,0], sample[:,1], time_set, label="original data")
        ax1.plot(prediction[:-3,0], prediction[:-3,1], time_set[:-2], label="prediction")
        ax1.legend()

        # Plotting dissipation potential
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(projection="3d")
        ax3.set_xlabel("c1*")
        ax3.set_ylabel("c2*")
        ax3.set_title("Dissipation potential Ξ(0.2, x*)")

        x1_star = torch.linspace(-1,1,500, dtype=torch.float32)
        x2_star = torch.linspace(-1,1,500, dtype=torch.float32)

        X1_star, X2_star = torch.meshgrid(x1_star, x2_star, indexing="ij")
        X1_star_flat = X1_star.flatten()
        X2_star_flat = X2_star.flatten()
        points = torch.stack([X1_star_flat, X2_star_flat], dim=1)

        dummy_x_input = torch.zeros_like(points, dtype=torch.float32) + 0.2

        Xi_flat = model(dummy_x_input, points)
        Xi_predicted = Xi_flat.reshape(X1_star.shape).cpu().detach().numpy()

        X1_star_np = X1_star.cpu().numpy()
        X2_star_np = X2_star.cpu().numpy()

        X = torch.matmul(points, -stoichiometric_matrix)

        Xi_analytic = 0
        for l in range(stoichiometric_matrix.shape[1]):
            W_l = 1
            for q in range(stoichiometric_matrix.shape[0]):
                W_l *= torch.sqrt((1e-7 + dummy_x_input[...,q]) ** abs(stoichiometric_matrix[q,l]))

            Xi_analytic += 2*W_l * (torch.exp(X[...,l]/2) + torch.exp(-X[...,l]/2) - 2)

        Xi_analytic = Xi_analytic.reshape(X1_star.shape).cpu()

        ax3.plot_surface(X1_star_np, X2_star_np, Xi_predicted, label="leared")
        ax3.plot_surface(X1_star_np, X2_star_np, Xi_analytic, label="analytic")
        ax3.legend()

    else:
        # Sampling random trajectory and plotting it along with predicted trajectory
        fig1, axes1 = plt.subplots(2, int(np.ceil(DIMENSION / 2)))
        axes1 = axes1.flatten()

        sample = test_pos[np.random.randint(0,len(test_pos)-1)].cpu().detach().numpy()
        tensor_sample = torch.tensor([sample], requires_grad=True)
        time_set = [args.dt*i for i in range(len(sample))]

        prediction = [sample[0]]
        print("calculating sample trajectory... it shouldn't take too long")
        for i in range(len(sample)):
            velocity = rk4(model.predict, torch.tensor(prediction[i], requires_grad=True), args.dt)
            prediction.append(prediction[i] + args.dt * velocity.cpu().detach().numpy())

        prediction = np.array(prediction)

        for d in range(DIMENSION):
            graph = axes1[d]

            graph.plot(time_set, sample[:,d], label="original data")
            graph.plot(time_set, prediction[:-1,d], label="prediction")

            graph.set_title(f"Species {d+1}")
            graph.set_xlabel("t")
            graph.set_ylabel("c")
            graph.legend()

        for i in range(DIMENSION, len(axes1)):
            fig1.delaxes(axes1[i])

        # Plotting dissipation potential
        fig3, axes3 = plt.subplots(2, int(np.ceil(DIMENSION / 2)))
        axes3 = axes3.flatten()

        for d in range(DIMENSION):
            graph = axes3[d]
            x_inputs = torch.full((500, DIMENSION), 0.2, dtype=torch.float32)
            x_star_inputs = torch.full((500, DIMENSION), 0, dtype=torch.float32)

            x_star = torch.linspace(-1,1,500, dtype=torch.float32)
            x_star_inputs[:,d] = x_star
            Xi_predicted = model(x_inputs, x_star_inputs).cpu().detach()

            X = torch.matmul(x_star_inputs, -stoichiometric_matrix)

            Xi_analytic = 0
            for l in range(stoichiometric_matrix.shape[1]):
                W_l = 1
                for q in range(stoichiometric_matrix.shape[0]):
                    W_l *= torch.sqrt((1e-7 + x_inputs[...,q]) ** abs(stoichiometric_matrix[q,l]))

                Xi_analytic += 2*W_l * (torch.exp(X[...,l]/2) + torch.exp(-X[...,l]/2) - 2)

            graph.plot(x_star.cpu(), Xi_predicted, label="learned")
            graph.plot(x_star.cpu(), Xi_analytic.cpu().numpy(), label="analytic")

            graph.set_xlabel(f"x*_{d+1}")
            graph.set_ylabel("Ξ")
            graph.legend()

        for i in range(DIMENSION, len(axes3)):
            fig3.delaxes(axes3[i])

    plt.show()
