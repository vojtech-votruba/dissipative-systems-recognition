import argparse
import numpy as np
import os
from matplotlib import pyplot as plt

"""
    Code for generating simple trajectories in phase space, x denotes position, p denotes momentum.
    c and b are the coefficients for our damped linear harmonic oscilator. m is the mass.
"""

parser = argparse.ArgumentParser(prog='simulate_dataset.py',
                                description='A short script for generating linear harmonic oscilator data used in a machine learning project.',)

parser.add_argument("--num" , default=100, type=int, help="number of trajectories that we want to simulate")
parser.add_argument("--points", default=1000, type=int, help="number of points for each particle trajectory")
parser.add_argument("--k", default=1.0, type=float, help= "stiffness coefficient")
parser.add_argument("--c", default=-0.3, type=float, help= "viscous damping coefficient")
parser.add_argument("--m", default=1.0, type=float, help= "mass of the particle")
parser.add_argument("--dim", default=1, type=int, help="spatial dimension of the phase space")
parser.add_argument("--dt", default=0.02, type=float, help="size of the time step used in the simulation")
parser.add_argument("--verbose", default=True, type=bool, help="print progress")
parser.add_argument("--plot", default=True, type=bool, help="plot the results")
args = parser.parse_args()

data = []

for n in range(args.num):
    x = np.array([100*np.random.random() for i in range(args.dim)])
    p = np.array([100*np.random.random() for i in range(args.dim)])
    time = 0
    dataset = []
    for i in range(args.points):
        dataset.append([time] + [each for each in x] + [each for each in p])
        a = -args.k * x + args.c * p/args.m
        p += (args.dt * a) * args.m
        x += p/args.m * args.dt + 0.5 * a * args.dt ** 2
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
    