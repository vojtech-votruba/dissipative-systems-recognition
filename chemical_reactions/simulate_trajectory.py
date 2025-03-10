import argparse
import numpy as np
import random
import os
from matplotlib import pyplot as plt

"""
    Code for generating chemical reaction trajectories in the spirit of gradient dynamics.
    The method of numerical integration we leverage is the standard 4th order Runge Kutta.
"""

parser = argparse.ArgumentParser(prog='simulate_trajectory.py',
                                description='A short script for generating gradient dynamics data used in a machine learning project.',)

parser.add_argument("--num" , default=128, type=int, help="number of trajectories that we want to simulate")
parser.add_argument("--points", default=4096, type=int, help="number of points for each trajectory")
parser.add_argument("--dt", default=0.003, type=float, help="size of the time step used in the simulation")
parser.add_argument("--verbose", default=True, type=bool, help="print progress")
parser.add_argument("--plot", default=True, type=bool, help="plot the results")
parser.add_argument("--gamma", default=1.0, type=float, help="the speed constant")
args = parser.parse_args()

def evolution(x):
    """
        We are assuming the entropy to be something like S_i = -c_i (ln(c_i) - 1),
        and the dissipation potential to be in the form gamma*sqrt(c1c2) cos((c1*-c2*) / 2) - gamma*sqrt(c1c2)
    """
    x_dot[0] = args.gamma*1/4 * (x[1] - x[0])
    x_dot[1] = args.gamma*1/4 * (x[0] - x[1])

    return x_dot

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
np.random.seed(42)

for n in range(args.num):
    x = np.array([random.uniform(0,1), random.uniform(0,1)])
    x_dot = np.array([0, 0])
    
    time = 0
    dataset = []
    for i in range(args.points):
        dataset.append([time] + [each for each in x] + [each for each in x_dot])

        x_dot = rk4(evolution, x, args.dt)
        x += x_dot * args.dt
        time += args.dt

    dataset = np.array(dataset)
    data.append(dataset)

    if args.verbose:
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

if args.verbose:
    print("\nDone! Trajectories saved into ./data/dataset.txt")

if args.plot:
    plt.style.use("bmh")
    custom_cycler = plt.cycler(
        color=['#6c3b9c', '#a02c2c', '#2c7a2c', '#2c2c7a', '#287d7d', '#aa5500', '#555555'],
        linestyle=['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
    )
    plt.rc('axes', prop_cycle=custom_cycler)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    random_trajectory = data[np.random.randint(0,args.num-1)]

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("t")
    ax.plot(random_trajectory[:,1], random_trajectory[:,2], random_trajectory[:,0], label="original data")

    plt.show()
    
else:
    raise Exception("Plotting is not supported for different dimensions than 2.")
    