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

parser.add_argument("--epochs", default=800, type=int, help="number of epoches for the model to train")
parser.add_argument("--batch_size", default=128, type=int, help="batch size for training of the model")
parser.add_argument("--dt", default=0.003, type=float, help="size of the time step used in the simulation")
parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help="do you wish to train a new model?")
parser.add_argument('--plot', default=True, action=argparse.BooleanOptionalAction, help="option of plotting the loss function")
parser.add_argument("--log", default=True, type=int, help="using log loss for plotting and such")
parser.add_argument("--eps", default=5.0, type=float, help="small epsilon used for weights reparametrization")
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
    return -1 * x

class DissipationNetwork(nn.Module):
    """
        For this network we are using a more complex architecture to ensure 
        only a partial convexity of the output with respect to some inputs.
        Specifically: PICNN, source: https://arxiv.org/pdf/1609.07152
    """
    def __init__(self):
        super().__init__()
        # The branch that propagates x directly forward
        self.x_input_layer = nn.Linear(DIMENSION, 3)
        self.x_prop_layer1 = nn.Linear(3, 3)
        self.x_prop_layer2 = nn.Linear(3, 3)

        # The branch that goes directly between x and x_star
        self.x_lateral_layer_1 = nn.Linear(DIMENSION, 3)
        self.x_lateral_layer_2 = nn.Linear(3, 3)
        self.x_lateral_layer_3 = nn.Linear(3, 3)
        self.x_lateral_layer_out = nn.Linear(3, 1)

        # The branch that propagates x_star forward (We need to enforce convexity here)
        self.conjugate_prop_layer_1 = PositiveLinear(3, 3, bias=False)
        self.conjugate_prop_layer_2 = PositiveLinear(3, 3, bias=False)
        self.conjugate_prop_layer_out= PositiveLinear(3, 1, bias=False)

        self.conjugate_prop_layer_1_mid = nn.Linear(3, 3)
        self.conjugate_prop_layer_2_mid = nn.Linear(3, 3)
        self.conjugate_prop_layer_out_mid = nn.Linear(3, 3)

        # The branch which always starts at x0_star and ends at arbitrary x_star
        self.conjugate_lateral_layer_in = nn.Linear(DIMENSION, 3, bias=False)
        self.conjugate_lateral_layer_1 = nn.Linear(DIMENSION, 3, bias=False)
        self.conjugate_lateral_layer_2 = nn.Linear(DIMENSION, 3, bias=False)
        self.conjugate_lateral_layer_out = nn.Linear(DIMENSION, 1, bias=False)

        self.conjugate_lateral_layer_in_mid = nn.Linear(DIMENSION, DIMENSION)
        self.conjugate_lateral_layer_1_mid = nn.Linear(3, DIMENSION)
        self.conjugate_lateral_layer_2_mid = nn.Linear(3, DIMENSION)
        self.conjugate_lateral_layer_out_mid = nn.Linear(3, DIMENSION)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input, input_star):
        x0 = input
        x0_star = input_star

        x_star = nn.Softplus()(self.x_lateral_layer_1(x0) 
                               + self.conjugate_lateral_layer_in(torch.mul(x0_star, self.conjugate_lateral_layer_in_mid(x0))))
        x = nn.Softplus()(self.x_input_layer(x0))

        x_star = nn.Softplus()(self.x_lateral_layer_2(x) 
                               + self.conjugate_prop_layer_1(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_1_mid(x))))
                                + self.conjugate_lateral_layer_1(torch.mul(x0_star, self.conjugate_lateral_layer_1_mid(x))))
        x = nn.Softplus()(self.x_prop_layer1(x))

        x_star = nn.Softplus()(self.x_lateral_layer_3(x) 
                               + self.conjugate_prop_layer_2(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_2_mid(x))))
                                + self.conjugate_lateral_layer_2(torch.mul(x0_star, self.conjugate_lateral_layer_2_mid(x))))
        x = nn.Softplus()(self.x_prop_layer2(x))

        out = nn.Softplus()(self.x_lateral_layer_out(x) 
                            + self.conjugate_prop_layer_out(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_out_mid(x))))\
                                + self.conjugate_lateral_layer_out(torch.mul(x0_star, self.conjugate_lateral_layer_out_mid(x))))

        return out

class GradientDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.Psi = DissipationNetwork()

    def forward(self, x):
        x = x.float()
        x_star = conjugate(x)

        Psi = self.Psi(x, x_star)
        x_dot = autograd.grad(Psi, x_star, grad_outputs=torch.ones_like(Psi), create_graph=True)[0]

        return x_dot

L = nn.MSELoss()

if args.train:
    training_trajectories, test_trajectories = random_split(trajectories, [0.8, 0.2], generator=generator)
    dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size, shuffle=True, generator=generator)

    model = GradientDynamics().to(DEVICE)

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-3, max_iter=10, history_size=20, line_search_fn='strong_wolfe')

    # Training
    trajectory_losses = []
    velocity_losses = []
    conservation_losses = []
    integrability_losses = []
    origin_losses = []
    minimum_losses = []

    loss_scales = None
    alpha = 0.95
    epsilon = 1e-8

    for i in range(args.epochs):
        for j, (pos, veloc, targ_pos, targ_veloc) in enumerate(dataloader):
            pos = pos.to(DEVICE)
            veloc = veloc.to(DEVICE)
            targ_pos = targ_pos.to(DEVICE)
            targ_veloc = targ_veloc.to(DEVICE)

            if i < args.epochs // 1.1:
                optimizer = adam_optimizer
                optimizer.zero_grad()

                predicted_veloc = rk4(model, pos, args.dt)
                trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos)
                velocity_loss = L(predicted_veloc, veloc)
                pos_star = conjugate(pos)

                Psi_array = model.Psi(pos, pos_star)
                dPsi = Psi_array[:, 1:, :] - Psi_array[:, :-1, :]
                conservation_loss = L(dPsi, torch.zeros_like(dPsi))

                Psi_origin_array = model.Psi(pos, torch.zeros_like(pos))
                origin_loss = L(Psi_origin_array, torch.zeros_like(Psi_origin_array))

                zeros_star = torch.zeros_like(pos, requires_grad=True)
                dPsi_dx_star_zero = autograd.grad(model.Psi(pos, zeros_star), zeros_star, grad_outputs=torch.ones_like(Psi_array), create_graph=True, retain_graph=True)[0]
                minimum_loss = L(dPsi_dx_star_zero, torch.zeros_like(dPsi_dx_star_zero))

                if loss_scales is None:
                    loss_scales = {
                        "trajectory": trajectory_loss.item() + epsilon,
                        "velocity": velocity_loss.item() + epsilon,
                        "conservation": conservation_loss.item() + epsilon,
                        "origin": origin_loss.item() + epsilon,
                        "minimum": minimum_loss.item() + epsilon,
                    }
                else:
                    with torch.no_grad():
                        loss_scales["trajectory"] = alpha * loss_scales["trajectory"] + (1 - alpha) * (trajectory_loss.item() + epsilon)
                        loss_scales["velocity"] = alpha * loss_scales["velocity"] + (1 - alpha) * (velocity_loss.item() + epsilon)
                        loss_scales["conservation"] = alpha * loss_scales["conservation"] + (1 - alpha) * (conservation_loss.item() + epsilon)
                        loss_scales["origin"] = alpha * loss_scales["origin"] + (1 - alpha) * (origin_loss.item() + epsilon)
                        loss_scales["minimum"] = alpha * loss_scales["minimum"] + (1 - alpha) * (minimum_loss.item() + epsilon)

                loss = (
                    0.1 * trajectory_loss / loss_scales["trajectory"] +
                    0.6 * velocity_loss / loss_scales["velocity"] +
                    0.1 * conservation_loss / loss_scales["conservation"] +
                    0.1 * origin_loss / loss_scales["origin"] + 
                    0.1 * minimum_loss / loss_scales["minimum"]
                )
                loss.backward()

                optimizer.step()
                
            else:
                optimizer = lbfgs_optimizer

                def closure():
                    optimizer.zero_grad()

                    predicted_veloc = rk4(model, pos, args.dt)
                    trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos)
                    velocity_loss = L(predicted_veloc, veloc)
                    pos_star = conjugate(pos)

                    Psi_array = model.Psi(pos, pos_star)
                    dPsi = Psi_array[:, 1:, :] - Psi_array[:, :-1, :]
                    conservation_loss = L(dPsi, torch.zeros_like(dPsi))

                    Psi_origin_array = model.Psi(pos, torch.zeros_like(pos))
                    origin_loss = L(Psi_origin_array, torch.zeros_like(Psi_origin_array))

                    zeros_star = torch.zeros_like(pos, requires_grad=True)
                    dPsi_dx_star_zero = autograd.grad(model.Psi(pos, zeros_star), zeros_star, grad_outputs=torch.ones_like(Psi_array), create_graph=True, retain_graph=True)[0]
                    minimum_loss = L(dPsi_dx_star_zero, torch.zeros_like(dPsi_dx_star_zero))

                    loss = (
                        0.1 * trajectory_loss / loss_scales["trajectory"] +
                        0.6 * velocity_loss / loss_scales["velocity"] +
                        0.1 * conservation_loss / loss_scales["conservation"] +
                        0.1 * origin_loss / loss_scales["origin"] + 
                        0.1 * minimum_loss / loss_scales["minimum"]
                    )
                    loss.backward()
                    
                    return loss

                optimizer.step(closure)

            predicted_veloc = rk4(model, pos, args.dt)
            trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos)
            velocity_loss = L(predicted_veloc, veloc)
            pos_star = conjugate(pos)

            Psi_array = model.Psi(pos, pos_star)
            dPsi = Psi_array[:, 1:, :] - Psi_array[:, :-1, :]
            conservation_loss = L(dPsi, torch.zeros_like(dPsi))

            dPsi_dx = autograd.grad(Psi_array, pos, grad_outputs=torch.ones_like(Psi_array), create_graph=True)[0]
            dPsi_dx_star = autograd.grad(Psi_array, pos_star, grad_outputs=torch.ones_like(Psi_array), create_graph=True)[0]
            dPsi_dx_star_dx = autograd.grad(dPsi_dx, pos_star, grad_outputs=torch.ones_like(dPsi_dx), create_graph=True)[0]
            dPsi_dx_dx_star = autograd.grad(dPsi_dx_star, pos, grad_outputs=torch.ones_like(dPsi_dx_star), create_graph=True)[0]
            integrability_loss = L(dPsi_dx_star_dx, dPsi_dx_dx_star)

            Psi_origin_array = model.Psi(pos, torch.zeros_like(pos))
            origin_loss = L(Psi_origin_array, torch.zeros_like(Psi_origin_array))

            minimum_loss = L(torch.min(Psi_array, torch.zeros_like(Psi_array)), torch.zeros_like(Psi_array))

            if args.log:
                trajectory_losses.append(np.log(trajectory_loss.item()))
                velocity_losses.append(np.log(velocity_loss.item()))
                conservation_losses.append(np.log(conservation_loss.item()))
                integrability_losses.append(np.log(integrability_loss.item()))
                origin_losses.append(np.log(origin_loss.item()))
                minimum_losses.append(np.log(minimum_loss.item()))

            else:
                trajectory_losses.append(trajectory_loss.item())
                velocity_losses.append(velocity_loss.item())
                conservation_losses.append(conservation_loss.item())
                integrability_loss.append(integrability_loss.item())
                origin_losses.append(origin_loss.item())
                minimum_losses.append(minimum_loss.item())

        print(f"Epoch no. {i}/{args.epochs} done! Traj. loss: {trajectory_loss}. Vel. loss: {velocity_loss}. Cons. loss: {conservation_loss}. Orig. loss: {conservation_loss}. Min. loss: {minimum_loss}")

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

MSE_trajectory_loss = L(model(test_pos) * args.dt + test_pos, test_target_pos)
MSE_velocity_loss = L(model(test_pos), test_vel)
MSE_test_set = L(model(test_pos) * args.dt + test_pos, test_target_pos) + L(model(test_pos), test_vel)
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
        ax0.plot(range(len(conservation_losses)), conservation_losses, label="conservation loss")
        ax0.plot(range(len(origin_losses)), origin_losses, label="origin loss")
        ax0.plot(range(len(minimum_losses)), minimum_losses, label="minimum loss")

        ax0.legend()

    if DIMENSION == 1:
        # Sampling random trajectory and plotting it along with predicted trajectory
        fig1,ax1 = plt.subplots()
        sample = test_pos[np.random.randint(0,len(test_pos)-1)].cpu().detach().numpy()
        tensor_sample = torch.tensor([sample], requires_grad=True)
        time_set = [args.dt*i for i in range(len(sample))]
    
        ax1.set_xlabel("t")
        ax1.set_ylabel("x")
        
        velocities = rk4(model, tensor_sample, args.dt)

        prediction = [sample[0]]
        for i in range(len(sample)):
            prediction.append(prediction[i] + args.dt * velocities[0][i].cpu().detach().numpy())

        prediction = np.array(prediction)
        ax1.set_title(f"Total MSE of the test set: {MSE_test_set}")

        ax1.plot(time_set[:-2], prediction[:-3] , label="prediction")
        ax1.plot(time_set, sample, label="original data")
        ax1.legend()

        # Plotting the dissipation potential, along our trajectory
        fig2,ax2 = plt.subplots()
        ax2.set_xlabel("t")
        ax2.set_ylabel("Ψ")

        sample_x_star = conjugate(tensor_sample)
        potential_evolution = model.Psi(tensor_sample, sample_x_star).squeeze(-1).squeeze(0).cpu().detach().numpy()

        ax2.plot(time_set, potential_evolution, label="learned")
        ax2.set_title(f"Dissipation potential in time, along the given trajectory")
        ax2.legend()

        # Plotting the learned dissipation potential
        fig3,ax3 = plt.subplots()
        ax3.set_xlabel("x*")
        ax3.set_ylabel("Ψ")
        ax3.set_title("Dissipation potential Ψ = Ψ(x=0, x*)")

        x_star_range = torch.linspace(-1,1,500, dtype=torch.float32).reshape(-1,1)
        zeros_column = torch.zeros_like(x_star_range, dtype=torch.float32).reshape(-1,1)

        ax3.plot(x_star_range.cpu(), model.Psi(zeros_column, x_star_range).cpu().detach(), label="learned")
        ax3.plot(x_star_range.cpu(), 1/2 * x_star_range.cpu()**2, label="analytic")
        ax3.legend()

#-------------------------------------------------------------------------------------------------------------------------------------

        fig5 = plt.figure()
        ax5 = fig5.add_subplot(projection="3d")
        ax5.set_xlabel("x")
        ax5.set_ylabel("x*")

        x_range = torch.linspace(-1, 1, 500, dtype=torch.float32).reshape(-1, 1)
        x_star_range = torch.linspace(-1, 1, 500, dtype=torch.float32).reshape(-1, 1)
        X, X_star = torch.meshgrid(x_range.squeeze(), x_star_range.squeeze(), indexing="ij")
        X_flat = X.flatten().reshape(-1, 1)
        X_star_flat = X_star.flatten().reshape(-1, 1)

        Psi_flat = model.Psi(X_flat, X_star_flat)
        Psi = Psi_flat.reshape(X.shape)

        X1_star_np = X.cpu().numpy()
        X2_star_np = X_star.cpu().numpy()
        Psi_np = Psi.cpu().detach().numpy()

        ax5.set_title("Dissipation potential Ψ(x, x*)")
        ax5.plot_surface(X1_star_np, X2_star_np, Psi_np)

#-------------------------------------------------------------------------------------------------------------------------------------

    if DIMENSION == 2:
        # Sampling random trajectory and plotting it along with predicted trajectory
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection="3d")
        sample = test_pos[np.random.randint(0,len(test_pos)-1)].cpu().detach().numpy()
        tensor_sample = torch.tensor([sample], requires_grad=True)
        time_set = [args.dt*i for i in range(len(sample))]

        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_zlabel("t")

        ax1.plot(sample[:,0], sample[:,1], time_set, label="original data")
        velocities = model(torch.tensor([sample], requires_grad=True))

        prediction = [sample[0]]

        for i in range(len(sample)):
            prediction.append(prediction[i] + args.dt * velocities[0][i].cpu().detach().numpy())

        ax1.set_title(f"MSE of the test set: {MSE_test_set}")
        prediction = np.array(prediction)

        ax1.plot(prediction[:-3,0], prediction[:-3,1], time_set[:-2], label="prediction")
        ax1.legend()

        # Plotting the dissipation potential, along our trajectory
        fig2,ax2 = plt.subplots()
        ax2.set_xlabel("t")
        ax2.set_ylabel("Ψ")

        sample_x_star = conjugate(tensor_sample)
        potential_evolution = model.Psi(tensor_sample, sample_x_star).squeeze(-1).squeeze(0).cpu().detach().numpy()

        ax2.plot(time_set, potential_evolution, label="learned")

        ax2.set_title(f"Dissipation potential in time, along the given trajectory")
        ax2.legend()

        # Plotting dissipation potential
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(projection="3d")
        ax3.set_xlabel("x1*")
        ax3.set_ylabel("x2*")

        x1_star = torch.linspace(-1,1,500, dtype=torch.float32)
        x2_star = torch.linspace(-1,1,500, dtype=torch.float32)

        X1_star, X2_star = torch.meshgrid(x1_star, x2_star, indexing="ij")
        X1_star_flat = X1_star.flatten()
        X2_star_flat = X2_star.flatten()
        points = torch.stack([X1_star_flat, X2_star_flat], dim=1)

        zeros_column = torch.zeros_like(points, dtype=torch.float32)

        Psi_flat = model.Psi(zeros_column, points)
        Psi = Psi_flat.reshape(X1_star.shape)

        X1_star_np = X1_star.cpu().numpy()
        X2_star_np = X2_star.cpu().numpy()
        Psi_np = Psi.cpu().detach().numpy()
        ax3.set_title("Dissipation potential Ψ(0, x*)")
        Psi_theor = 0.5 * (X1_star_np ** 2 + X2_star_np**2)

        ax3.plot_surface(X1_star_np, X2_star_np, Psi_np, label="leared")
        ax3.plot_surface(X1_star_np, X2_star_np, Psi_theor , label="analytic")
        ax3.legend()

    plt.show()
