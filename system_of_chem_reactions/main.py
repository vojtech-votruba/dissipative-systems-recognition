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

parser.add_argument("--epochs", default=1500, type=int, help="number of epoches for the model to train")
parser.add_argument("--batch_size", default=128, type=int, help="batch size for training of the model")
parser.add_argument("--dt", default=0.006, type=float, help="size of the time step used in the simulation")
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

class NegativeLinear(nn.Linear):
    """
        A positive layer that we use to enforce convexity
    """

    def forward(self, input):
        W = self.weight
        eps_tensor = torch.tensor(args.eps, device=W.device, dtype=W.dtype)

        positive_W = -torch.exp(-W - eps_tensor) 
        negative_W = W - torch.exp(-eps_tensor)
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

class EntropyNetwork(nn.Module):
    """
        For the entropy network we are using a fully input concave neural network achitecture,
        it's a simple alteration of FICNN - fully input convex neural nets,
        we just need to use concave, decreasing activation functions and negative weights instead of positive ones.
    """
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(DIMENSION, 5)

        self.prop_layer1 = NegativeLinear(5, 5)
        self.lateral_layer1 = nn.Linear(DIMENSION, 5)

        self.prop_layer2 = NegativeLinear(5, 5)
        self.lateral_layer2 = nn.Linear(DIMENSION, 5)

        self.output_layer = NegativeLinear(5, 1)
        self.lateral_layer_out = nn.Linear(DIMENSION, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x0):
        x = -nn.Softplus()(self.input_layer(x0))

        x = -nn.Softplus()(self.prop_layer1(x) + self.lateral_layer1(x0))

        x = -nn.Softplus()(self.prop_layer2(x) + self.lateral_layer2(x0))

        S = -nn.Softplus()(self.output_layer(x) + self.lateral_layer_out(x0))

        return S

class DissipationNetwork(nn.Module):
    """
        For this network we are using a more complex architecture to ensure 
        only a partial convexity of the output with respect to some inputs.
        Specifically: PICNN, source: https://arxiv.org/pdf/1609.07152
    """
    def __init__(self):
        super().__init__()
        # The branch that propagates x directly forward
        self.x_input_layer = nn.Linear(DIMENSION, 4)
        self.x_prop_layer1 = nn.Linear(4, 4)
        self.x_prop_layer2 = nn.Linear(4, 4)

        # The branch that goes directly between x and x_star
        self.x_lateral_layer_1 = nn.Linear(DIMENSION, 4)
        self.x_lateral_layer_2 = nn.Linear(4, 4)
        self.x_lateral_layer_3 = nn.Linear(4, 4)
        self.x_lateral_layer_out = nn.Linear(4, 1)

        # The branch that propagates x_star forward (We need to enforce convexity here)
        self.conjugate_prop_layer_1 = PositiveLinear(4, 4, bias=False)
        self.conjugate_prop_layer_2 = PositiveLinear(4, 4, bias=False)
        self.conjugate_prop_layer_out= PositiveLinear(4, 1, bias=False)

        self.conjugate_prop_layer_1_mid = nn.Linear(4, 4)
        self.conjugate_prop_layer_2_mid = nn.Linear(4, 4)
        self.conjugate_prop_layer_out_mid = nn.Linear(4, 4)

        # The branch which always starts at x0_star and ends at arbitrary x_star
        self.conjugate_lateral_layer_in = nn.Linear(DIMENSION, 4, bias=False)
        self.conjugate_lateral_layer_1 = nn.Linear(DIMENSION, 4, bias=False)
        self.conjugate_lateral_layer_2 = nn.Linear(DIMENSION, 4, bias=False)
        self.conjugate_lateral_layer_out = nn.Linear(DIMENSION, 1, bias=False)

        self.conjugate_lateral_layer_in_mid = nn.Linear(DIMENSION, DIMENSION)
        self.conjugate_lateral_layer_1_mid = nn.Linear(4, DIMENSION)
        self.conjugate_lateral_layer_2_mid = nn.Linear(4, DIMENSION)
        self.conjugate_lateral_layer_out_mid = nn.Linear(4, DIMENSION)

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
        self.S = EntropyNetwork()
        self.Xi = DissipationNetwork()

    def forward(self, x, x_star):
        x_star_zeros = torch.zeros_like(x, requires_grad=True)

        Xi_raw = self.Xi(x, x_star)
        Xi_at_zero = self.Xi(x, x_star_zeros)
        Xi = Xi_raw - Xi_at_zero - (x_star * autograd.grad(Xi_at_zero, x_star_zeros, grad_outputs=torch.ones_like(Xi_at_zero), create_graph=True)[0]).sum(dim=-1).unsqueeze(-1)

        return Xi
    
    def predict(self, x):
        S = self.S(x)
        x_star = autograd.grad(S, x, grad_outputs=torch.ones_like(S), create_graph=True)[0].float()
        Xi = self.forward(x,x_star)

        x_dot = autograd.grad(Xi, x_star, grad_outputs=torch.ones_like(Xi), create_graph=True)[0]
        return x_dot

L = nn.MSELoss()

if args.train:
    training_trajectories, test_trajectories = random_split(trajectories, [0.8, 0.2], generator=generator)

    lbfgs_dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size, shuffle=True, generator=generator)
    adam_dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size // 2, shuffle=True, generator=generator)

    model = GradientDynamics().to(DEVICE)

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-3, max_iter=10, history_size=20, line_search_fn='strong_wolfe')

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

                loss = (
                    trajectory_loss +
                    velocity_loss
                )
                loss.backward()

                optimizer.step()
            else:
                def closure():
                    optimizer.zero_grad()

                    predicted_veloc = rk4(model.predict, pos, args.dt)
                    trajectory_loss = L(predicted_veloc * args.dt + pos, targ_pos) / torch.std(targ_pos)
                    velocity_loss = L(predicted_veloc, veloc) / torch.std(veloc)

                    loss = (
                        trajectory_loss +
                        velocity_loss
                    )
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

        prediction = [sample[0]]
        print("calculating sample trajectory... it shouldn't take too long")
        for i in range(len(sample)):
            velocity = rk4(model.predict, torch.tensor(prediction[i], requires_grad=True), args.dt)
            prediction.append(prediction[i] + args.dt * velocity.cpu().detach().numpy())

        ax1.set_title(f"Sample trajectory")
        prediction = np.array(prediction)

        ax1.plot(sample[:,0], sample[:,1], time_set, label="original data")
        ax1.plot(prediction[:-3,0], prediction[:-3,1], time_set[:-2], label="prediction")
        ax1.legend()

        # Plotting dissipation potential
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(projection="3d")
        ax3.set_xlabel("c1*")
        ax3.set_ylabel("c2*")

        x1_star = torch.linspace(-1,1,500, dtype=torch.float32)
        x2_star = torch.linspace(-1,1,500, dtype=torch.float32)

        X1_star, X2_star = torch.meshgrid(x1_star, x2_star, indexing="ij")
        X1_star_flat = X1_star.flatten()
        X2_star_flat = X2_star.flatten()
        points = torch.stack([X1_star_flat, X2_star_flat], dim=1)

        zeros_column = torch.zeros_like(points, dtype=torch.float32) + 0.2

        Xi_flat = model(zeros_column, points)
        Xi = Xi_flat.reshape(X1_star.shape)

        X1_star_np = X1_star.cpu().numpy()
        X2_star_np = X2_star.cpu().numpy()
        Xi_np = Xi.cpu().detach().numpy()
        ax3.set_title("Dissipation potential Ξ(0.2, x*)")
        Xi_theor = 0.2 * 4 * (np.cosh((X1_star_np - X2_star_np) / 2) - 1)

        ax3.plot_surface(X1_star_np, X2_star_np, Xi_np, label="learned")
        ax3.plot_surface(X1_star_np, X2_star_np, Xi_theor , label="analytic")
        ax3.legend()

        # Plotting entropy
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(projection="3d")
        ax4.set_xlabel("x1")
        ax4.set_ylabel("x2")
        ax4.set_zlabel("S")

        x1 = torch.linspace(0.001,1,500, dtype=torch.float32)
        x2 = torch.linspace(0.001,1,500, dtype=torch.float32)

        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        X1_flat = X1.flatten()
        X2_flat = X2.flatten()
        points = torch.stack([X1_flat, X2_flat], dim=1)

        S_flat = model.S(points)
        S = S_flat.reshape(X1.shape)

        X1_np = X1.cpu().numpy()
        X2_np = X2.cpu().numpy()
        S_np = S.cpu().detach().numpy()

        S_theor = -X1_np * (np.log(X1_np) - 1) - X2_np * (np.log(X2_np) - 1)
        distance = S_theor - S_np

        ax4.plot_surface(X1_np, X2_np, S_np + distance, label="learned")
        ax4.plot_surface(X1_np, X2_np, S_theor, label="analytic")

        ax4.set_title("Entropy S = S(x1, x2)")
        ax4.legend()

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

        stoichiometric_matrix = torch.tensor([
        [-1, 1,],
        [1, -1,],])
    
        for d in range(DIMENSION):
            graph = axes3[d]
            x_inputs = torch.full((500, DIMENSION), 0.2, dtype=torch.float32)
            x_star_inputs = torch.full((500, DIMENSION), 0, dtype=torch.float32)

            x_star = torch.linspace(-1,1,500, dtype=torch.float32)
            x_star_inputs[:,d] = x_star
            Xi_predicted = model(x_inputs, x_star_inputs).cpu().detach().numpy()

            Xi_analytic = 0
            for l in range(stoichiometric_matrix.shape[1]):
                W_l = 1
                for q in range(stoichiometric_matrix.shape[0]):
                    W_l *= torch.sqrt(x_inputs[:,q] ** torch.abs(stoichiometric_matrix[q,l]))

                X_l = 0
                for q in range(stoichiometric_matrix.shape[0]):
                    X_l += x_star_inputs[:,q] * stoichiometric_matrix[q,l]

                Xi_analytic += W_l * (torch.exp(X_l/2) + torch.exp(-X_l/2) - 2)

            Xi_analytic = Xi_analytic.cpu().numpy()

            scaling = np.sum(Xi_analytic * Xi_predicted) / np.sum(Xi_analytic * Xi_analytic)
            scaling = 1
            
            graph.plot(x_star.cpu(), Xi_predicted/scaling, label="learned, scaled")
            graph.plot(x_star.cpu(), Xi_analytic, label="analytic")

            graph.set_xlabel(f"x*_{d+1}")
            #graph.set_ylabel(f"Ξ(0.2, 0.2, ..., 0, ..., x*_{d+1}, ..., 0)")
            graph.set_ylabel("Ξ")
            graph.legend()

        for i in range(DIMENSION, len(axes3)):
            fig3.delaxes(axes3[i])

        # Plotting entropy
        fig4, axes4 = plt.subplots(2, int(np.ceil(DIMENSION / 2)))
        axes4 = axes4.flatten()

        for d in range(DIMENSION):
            graph = axes4[d]
            x_inputs = torch.full((500, DIMENSION), 0.001, dtype=torch.float32)

            x_independent = torch.linspace(0.001,1,500, dtype=torch.float32)
            x_inputs[:,d] = x_independent
            S_predicted = model.S(x_inputs).cpu().detach().numpy()
            
            S_analytic = 0
            for i in range(DIMENSION):
                S_analytic -= x_inputs[:,i] * (torch.log(x_inputs[:,i]) - 1)
            S_analytic = S_analytic.cpu().numpy()

            distance = np.average(S_analytic - S_predicted)

            graph.plot(x_star.cpu(), S_predicted + distance, label="learned, shifted")
            graph.plot(x_star.cpu(), S_analytic, label="analytic")

            graph.set_xlabel(f"x_{d+1}")
            #graph.set_ylabel(f"Ξ(0.2, 0.2, ..., 0, ..., x*_{d+1}, ..., 0)")
            graph.set_ylabel("S")
            graph.legend()

        for i in range(DIMENSION, len(axes4)):
            fig4.delaxes(axes4[i])

    plt.tight_layout()
    plt.show()
