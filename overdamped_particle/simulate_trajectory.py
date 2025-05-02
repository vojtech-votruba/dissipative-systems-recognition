import argparse
import numpy as np
import os
from matplotlib import pyplot as plt
import scienceplots

"""
    Code for generating simple trajectories in the spirit of gradient dynamics.
    Both the dissipation potential, and entropy used in this simulation are quadratic, resulting
    in the equation of motion in the form \dot{x} = \gamma*x which we solve with rk4
"""

parser = argparse.ArgumentParser(prog='simulate_trajectory.py',
                                description='A short script for generating gradient dynamics data used in a machine learning project.',)

parser.add_argument("--num" , default=128, type=int, help="number of trajectories that we want to simulate")
parser.add_argument("--points", default=2048, type=int, help="number of points for each trajectory")
parser.add_argument("--dim", default=1, type=int, help="dimension of the data")
parser.add_argument("--dt", default=0.005, type=float, help="size of the time step used in the simulation")
parser.add_argument("--plot", default=True, type=bool, help="plot the results")
parser.add_argument("--gamma", default=1.0, type=float, help="the dampening constant")
args = parser.parse_args()

def evolution(x):
    """
        We are assuming the entropy to be something like S = -1/2 x**2
        and the dissipation potential to be in the form 1/2 * gamma * x_star ** 2
    """
    return -args.gamma * x

def rk4(f, x, time_step):
    """
        Classical 4th order Runge Kutta implementation,
    """

    k1i = f(x)
    k2i = f(x + k1i * time_step/2)
    k3i = f(x + k2i * time_step/2)
    k4i = f(x + k3i * time_step)

    return 1/6 * (k1i + 2*k2i + 2*k3i + k4i)

data = []
np.random.seed(21)

for n in range(args.num):
    x = np.array([np.random.uniform(-1,1) for i in range(args.dim)])
    
    time = 0
    dataset = []
    for i in range(args.points):
        dataset.append([time] + [each for each in x])

        x_dot = rk4(evolution, x, args.dt)
        x += x_dot * args.dt
        time += args.dt

    dataset = np.array(dataset)
    data.append(dataset)

    print(f"{n}/{args.num}", end='\r')

if os.path.exists("data"):
    os.remove("data/dataset.txt")
    for trajectory in data:
        with open("data/dataset.txt", "ab") as f:
            np.savetxt(f, trajectory, delimiter=",")
            f.write("\n".encode())
else:
    os.mkdir("data")    
    for trajectory in data:
        with open("data/dataset.txt", "ab") as f:
            np.savetxt(f, trajectory, delimiter=",")
            f.write("\n".encode())

print("\n Done! Trajectories saved into ./data/dataset.txt")

if args.plot:
    plt.style.use(['science','ieee'])
    
    if args.dim == 1:
        plt.xlabel(r"$t$")
        plt.ylabel(r"$x$")
        random_trajectory = data[np.random.randint(0,args.num-1)]
        plt.plot(random_trajectory[:,0], random_trajectory[:,1])
        plt.savefig("results/example.pdf")

        plt.show()
    
    elif args.dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        random_trajectory = data[np.random.randint(0,args.num-1)]

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$t$", labelpad=-4)
        ax.plot(random_trajectory[:,1], random_trajectory[:,2], random_trajectory[:,0], label="original data")

        plt.savefig("results/example2d.pdf")
        plt.show()
        
    else:
        raise Exception("Plotting is not supported for more dimensions than 1 and 2.")
    