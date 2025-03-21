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
parser.add_argument("--points", default=2048, type=int, help="number of points for each trajectory")
parser.add_argument("--dt", default=0.002, type=float, help="size of the time step used in the simulation")
parser.add_argument("--verbose", default=True, type=bool, help="print progress")
parser.add_argument("--plot", default=True, type=bool, help="plot the results")
parser.add_argument("--gamma", default=1.0, type=float, help="the speed constant")
args = parser.parse_args()

"""
Each row represents one species, each column one chemical reaction

    Example 1 (simplest reversible reaction):
    A <-> B

stoichiometric_matrix = np.array([
[-1.0],
[+1.0],])
                                
    Example 2 (water creation):
    2A + B <-> 2C

stoichiometric_matrix = np.array([
[-2.0],
[-1.0],
[+2.0],])

    Example 3 (a more complex reversible reaction):
    A + B <-> C
    C + D <-> B + E

stoichiometric_matrix = np.array([
[-1.0,+0.0],
[-1.0,+1.0],
[+1.0,-1.0],
[+0.0,-1.0],
[+0.0,+1.0],])
"""

stoichiometric_matrix = np.array([
[-1.0,+0.0],
[-1.0,+1.0],
[+1.0,-1.0],
[+0.0,-1.0],
[+0.0,+1.0],])

def evolution(x):
    """
        The evolution is calculated using mass action law.

        We are setting all forward and backward rates to 1
    """

    kinetics = np.ones(shape=stoichiometric_matrix.shape[1])
    for j in range(stoichiometric_matrix.shape[1]):
        forward = 1
        backward = 1

        for i in range(stoichiometric_matrix.shape[0]):
            if stoichiometric_matrix[i, j] < 0:
                forward *= x[i] ** abs(stoichiometric_matrix[i, j])
            elif stoichiometric_matrix[i, j] > 0:
                if x[i] > 0:
                    backward *= x[i] ** abs(stoichiometric_matrix[i, j])
                else:
                    backward = 0

        kinetics[j] = forward - backward
        
    x_dot = args.gamma * stoichiometric_matrix @ kinetics
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
    x = np.array([random.uniform(0.001,1) for j in range(stoichiometric_matrix.shape[0])])
    x_dot = np.array([0 for j in range(stoichiometric_matrix.shape[0])])
    
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

    if stoichiometric_matrix.shape[0] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        random_trajectory = data[np.random.randint(0,args.num-1)]

        ax.set_xlabel("c1")
        ax.set_ylabel("c2")
        ax.set_zlabel("t")
        ax.plot(random_trajectory[:,1], random_trajectory[:,2], random_trajectory[:,0], label="original data")

        plt.show()

    else:
        fig, ax = plt.subplots()
        random_trajectory = data[np.random.randint(0,args.num-1)]
        ax.set_xlabel("t")
        ax.set_ylabel("c")
    
        for i in range(stoichiometric_matrix.shape[0]):
            ax.plot(random_trajectory[:,0], random_trajectory[:,i+1], label=f"Species {i+1}")
        
        ax.legend()
    
        plt.show()
        