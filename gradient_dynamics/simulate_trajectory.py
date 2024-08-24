import argparse
import numpy as np
import os
from matplotlib import pyplot as plt

"""
    Code for generating simple trajectories in the spirit of gradient dynamics.
    Both the dissipation potential, and entropy used in this simulation are quadratic, resulting
    in the equation of motion in the form \dot{x} = \gamma*x which we solve with Euler's integration.
"""

parser = argparse.ArgumentParser(prog='simulate_trajectory.py',
                                description='A short script for generating gradient dynamics data used in a machine learning project.',)

parser.add_argument("--num" , default=100, type=int, help="number of trajectories that we want to simulate")
parser.add_argument("--points", default=1000, type=int, help="number of points for each trajectory")
parser.add_argument("--dim", default=1, type=int, help="dimension of the data")
parser.add_argument("--dt", default=0.02, type=float, help="size of the time step used in the simulation")
parser.add_argument("--verbose", default=True, type=bool, help="print progress")
parser.add_argument("--plot", default=True, type=bool, help="plot the results")
parser.add_argument("--gamma", default=-1.0, type=float, help="the dampening constant")
args = parser.parse_args()

data = []

for n in range(args.num):
    x = np.array([100*(0.5-np.random.random()) for i in range(args.dim)])
    x_dot = np.array([100*(0.5-np.random.random()) for i in range(args.dim)])
    time = 0
    dataset = []
    for i in range(args.points):
        dataset.append([time] + [each for each in x] + [each for each in x_dot])
        x += x_dot * args.dt
        x_dot = args.gamma * x
        time += args.dt

    dataset = np.array(dataset)
    data.append(dataset)

    if args.verbose:
            print(f"{n}/{args.num}")

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

if args.verbose:
    print("\nDone! Trajectories saved into ./data/dataset.txt")

if args.plot:
    if args.dim == 1:
        plt.xlabel("t")
        plt.ylabel("x")
        random_trajectory = data[np.random.randint(0,args.num-1)]
        plt.plot(random_trajectory[:,0], random_trajectory[:,1])
        plt.show()

    else:
        raise Exception("Plotting is not supported for more dimensions than 1.")
    