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

parser.add_argument("--epochs", default=400, type=int, help="number of epoches for the model to train")
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
        self.x_input_layer = nn.Linear(DIMENSION, 8)
        self.x_prop_layer1 = nn.Linear(8, 8)
        self.x_prop_layer2 = nn.Linear(8, 8)

        # The branch that goes directly between x and x_star
        self.x_lateral_layer_1 = nn.Linear(DIMENSION, 8)
        self.x_lateral_layer_2 = nn.Linear(8, 8)
        self.x_lateral_layer_3 = nn.Linear(8, 8)
        self.x_lateral_layer_out = nn.Linear(8, 1)

        # The branch that propagates x_star forward (We need to enforce convexity here)
        self.conjugate_prop_layer_1 = nn.Linear(8, 8, bias=False)
        self.conjugate_prop_layer_2 = nn.Linear(8, 8, bias=False)
        self.conjugate_prop_layer_out= nn.Linear(8, 1, bias=False)

        self.conjugate_prop_layer_1_mid = nn.Linear(8, 8)
        self.conjugate_prop_layer_2_mid = nn.Linear(8, 8)
        self.conjugate_prop_layer_out_mid = nn.Linear(8, 8)

        # The branch which always starts at x0_star and ends at arbitrary x_star
        self.conjugate_lateral_layer_in = nn.Linear(DIMENSION, 8, bias=False)
        self.conjugate_lateral_layer_1 = nn.Linear(DIMENSION, 8, bias=False)
        self.conjugate_lateral_layer_2 = nn.Linear(DIMENSION, 8, bias=False)
        self.conjugate_lateral_layer_out = nn.Linear(DIMENSION, 1, bias=False)

        self.conjugate_lateral_layer_in_mid = nn.Linear(DIMENSION, DIMENSION)
        self.conjugate_lateral_layer_1_mid = nn.Linear(8, DIMENSION)
        self.conjugate_lateral_layer_2_mid = nn.Linear(8, DIMENSION)
        self.conjugate_lateral_layer_out_mid = nn.Linear(8, DIMENSION)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
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
        self.conjugate_prop_layer_1.weight.data = enforce_pos(self.conjugate_prop_layer_1.weight.data)

        x_star = nn.Softplus()(self.x_lateral_layer_3(x) 
                               + self.conjugate_prop_layer_2(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_2_mid(x))))
                                + self.conjugate_lateral_layer_2(torch.mul(x0_star, self.conjugate_lateral_layer_2_mid(x))))
        x = nn.Softplus()(self.x_prop_layer2(x))
        self.conjugate_prop_layer_2.weight.data = enforce_pos(self.conjugate_prop_layer_2.weight.data)

        out = nn.Softplus()(self.x_lateral_layer_out(x) 
                            + self.conjugate_prop_layer_out(torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_out_mid(x))))\
                                + self.conjugate_lateral_layer_out(torch.mul(x0_star, self.conjugate_lateral_layer_out_mid(x))))
        self.conjugate_prop_layer_out.weight.data = enforce_pos(self.conjugate_prop_layer_out.weight.data)

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
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(42)
    training_trajectories, test_trajectories = random_split(trajectories, [0.8, 0.2], generator=generator)
    dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size, shuffle=True, generator=generator)

    model = GradientDynamics().to(DEVICE)
    """
        I have been trying multiple optimizers, according to one article I found LBFGS
        should be the best for PINNs.
    """
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3, amsgrad=True)
    lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-3, max_iter=10, history_size=15, line_search_fn='strong_wolfe')

    # Training
    losses = []
    trajectory_losses = []
    velocity_losses = []

    for i in range(args.epochs):
        for j, (pos, veloc, targ_pos, targ_veloc) in enumerate(dataloader):
            pos = pos.to(DEVICE)
            veloc = veloc.to(DEVICE)
            targ_pos = targ_pos.to(DEVICE)
            targ_veloc = targ_veloc.to(DEVICE)

            if i < args.epochs // 3:
                """
                    Firstly we try to find coarse convergence with Adam
                """
                optimizer = adam_optimizer
                optimizer.zero_grad()
                predicted_veloc = rk4(model, pos, args.dt)
                trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos)
                velocity_loss = L(predicted_veloc, veloc)

                loss = trajectory_loss + velocity_loss
                loss.backward()
                optimizer.step()
                
            else:
                """
                    Then we switch to a more precise second order optimizer
                """
                optimizer = lbfgs_optimizer
                
                def closure():
                    optimizer.zero_grad()
                    predicted_veloc = rk4(model, pos, args.dt)
                    trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos)
                    velocity_loss = L(predicted_veloc, veloc)
                    
                    loss = trajectory_loss + velocity_loss
                    loss.backward()
                    return loss
                
                optimizer.step(closure)
                
                predicted_veloc = rk4(model, pos, args.dt)
                trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos)
                velocity_loss = L(predicted_veloc, veloc)
                loss = trajectory_loss + velocity_loss

            if args.log:
                losses.append(np.log(loss.item()))
                trajectory_losses.append(np.log(trajectory_loss.item()))
                velocity_losses.append(np.log(velocity_loss.item()))
            else:
                losses.append(loss.item())
                trajectory_losses.append(trajectory_loss.item())
                velocity_losses.append(velocity_loss.item())

        print(f"Epoch no. {i}/{args.epochs} done! Trajectory loss: {trajectory_loss}. Velocity loss: {velocity_loss}")


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
    # Plotting the MSE decline
    if args.train:
        fig1,ax1 = plt.subplots()
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("MSE")
        if args.log:
            ax1.set_title("Total training log loss decline on the training data")
        else:
            ax1.set_title("Total training loss decline on the training data")
        ax1.plot(range(len(losses)), losses)

        fig2,ax2 = plt.subplots()
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("MSE")
        if args.log:
            ax2.set_title("Trajectory training log loss decline on the training data")
        else:
            ax2.set_title("Trajectory training loss decline on the training data")
        ax2.plot(range(len(trajectory_losses)), trajectory_losses)

        fig3,ax3 = plt.subplots()
        ax3.set_xlabel("Epochs")
        ax3.set_ylabel("MSE")
        if args.log:
            ax3.set_title("Velocity training log loss decline on the training data")
        else:
            ax3.set_title("Velocity training loss decline on the training data")
        ax3.plot(range(len(velocity_losses)), velocity_losses)

    if DIMENSION == 1:
        # Sampling random trajectory and plotting it along with predicted trajectory
        fig4,ax4 = plt.subplots()
        sample = test_pos[np.random.randint(0,len(test_pos)-1)].cpu().detach().numpy()
        tensor_sample = torch.tensor([sample], requires_grad=True)
        time_set = [args.dt*i for i in range(len(sample))]
    
        ax4.set_xlabel("t")
        ax4.set_ylabel("x")
        ax4.plot(time_set, sample, label="original data")
        velocities = rk4(model, tensor_sample, args.dt)

        prediction = [sample[0]]
        for i in range(len(sample)):
            prediction.append(prediction[i] + args.dt * velocities[0][i].cpu().detach().numpy())

        prediction = np.array(prediction)
        ax4.set_title(f"Total MSE of the test set: {MSE_test_set}")

        ax4.plot(time_set[:-2], prediction[:-3] , label="prediction")
        ax4.legend()

        # Plotting the dissipation potential, along our trajectory
        fig5,ax5 = plt.subplots()
        ax5.set_xlabel("t")
        ax5.set_ylabel("Ψ")

        sample_x_star = conjugate(tensor_sample)
        potential_evolution = model.Psi(tensor_sample, sample_x_star).squeeze(-1).squeeze(0).cpu().detach().numpy()

        ax5.plot(time_set, potential_evolution, label="learned")
        ax5.plot(time_set, np.zeros_like(time_set), label="analytical")

        ax5.set_title(f"Dissipation potential in time, along the given trajectory")
        ax5.legend()

        # Plotting the learned dissipation potential
        fig6,ax6 = plt.subplots()
        ax6.set_xlabel("x*")
        ax6.set_ylabel("Ψ")
        x_star_range = torch.linspace(-1,1,500, dtype=torch.float32).reshape(-1,1)
        zeros_column = torch.zeros_like(x_star_range, dtype=torch.float32).reshape(-1,1)

        ax6.plot(x_star_range.cpu(), model.Psi(zeros_column, x_star_range).cpu().detach(), label="learned")
        ax6.set_title("Dissipation potential Ψ = Ψ(x=0, x*)")
        ax6.plot(x_star_range.cpu(), 1/2 * x_star_range.cpu()**2, label="analytic")
        ax6.legend()

    if DIMENSION == 2:
        # Sampling random trajectory and plotting it along with predicted trajectory
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(projection="3d")
        sample = test_pos[np.random.randint(0,len(test_pos)-1)].cpu().detach().numpy()
        tensor_sample = torch.tensor([sample], requires_grad=True)
        time_set = [args.dt*i for i in range(len(sample))]

        ax4.set_xlabel("x1")
        ax4.set_ylabel("x2")
        ax4.set_zlabel("t")

        ax4.plot(sample[:,0], sample[:,1], time_set, label="original data")
        velocities = model(torch.tensor([sample], requires_grad=True))

        prediction = [sample[0]]

        for i in range(len(sample)):
            prediction.append(prediction[i] + args.dt * velocities[0][i].cpu().detach().numpy())

        ax4.set_title(f"MSE of the test set: {MSE_test_set}")
        prediction = np.array(prediction)

        ax4.plot(prediction[:-3,0], prediction[:-3,1], time_set[:-2], label="prediction")
        ax4.legend()

        # Plotting the dissipation potential, along our trajectory
        fig5,ax5 = plt.subplots()
        ax5.set_xlabel("t")
        ax5.set_ylabel("ln(Ψ)")

        sample_x_star = conjugate(tensor_sample)
        potential_evolution = np.log(model.Psi(tensor_sample, sample_x_star).squeeze(-1).squeeze(0).cpu().detach().numpy())

        ax5.plot(time_set, potential_evolution, label="learned")
        ax5.plot(time_set, np.zeros_like(time_set), label="analytical")

        ax5.set_title(f"Dissipation potential in time, along the given trajectory")
        ax5.legend()

        # Plotting dissipation potential
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(projection="3d")
        ax6.set_xlabel("x1*")
        ax6.set_ylabel("x2*")

        x1_star = torch.linspace(-1,1,500, dtype=torch.float32)
        x2_star = torch.linspace(-1,1,500, dtype=torch.float32)

        X1_star, X2_star = torch.meshgrid(x1_star, x2_star, indexing="ij")
        X1_star_flat = X1_star.flatten()
        X2_star_flat = X2_star.flatten()
        points = torch.stack([X1_star_flat, X2_star_flat], dim=1)

        ones_column = torch.zeros_like(points, dtype=torch.float32)

        Psi_flat = model.Psi(ones_column, points)
        Psi = Psi_flat.reshape(X1_star.shape)

        X1_star_np = X1_star.cpu().numpy()
        X2_star_np = X2_star.cpu().numpy()
        Xi_np = Psi.cpu().detach().numpy()
        ax6.set_title("Dissipation potential Ψ(0, x*)")
        Xi_theor = 0.5 * (X1_star_np ** 2 + X2_star_np**2)

        ax6.plot_surface(X1_star_np, X2_star_np, Xi_np, label="learned")
        ax6.plot_surface(X1_star_np, X2_star_np, Xi_theor , label="analytic")
        ax6.legend() 
        
    plt.show()
