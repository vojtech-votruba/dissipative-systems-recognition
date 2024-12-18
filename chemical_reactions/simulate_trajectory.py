import argparse
import numpy as np
import os
from matplotlib import pyplot as plt

"""
    Code for generating chemical reaction trajectories in the spirit of gradient dynamics.
    The method of numerical integration we leverage is the standard 4th order Runge Kutta.
"""

parser = argparse.ArgumentParser(prog='simulate_trajectory.py',
                                description='A short script for generating gradient dynamics data used in a machine learning project.',)

parser.add_argument("--num" , default=50, type=int, help="number of trajectories that we want to simulate")
parser.add_argument("--points", default=2000, type=int, help="number of points for each trajectory")
parser.add_argument("--dt", default=0.02, type=float, help="size of the time step used in the simulation")
parser.add_argument("--verbose", default=True, type=bool, help="print progress")
parser.add_argument("--plot", default=True, type=bool, help="plot the results")
parser.add_argument("--gamma", default=400.0, type=float, help="the speed constant")
args = parser.parse_args()

DIM = 2
def evolution(x):
    """
        We are assuming the entropy to be something like S_i = c_i R ln(c_i/c0),
        and the dissipation potential to be in a form 1/sqrt(c1c2) cos((c1*-c2*) / 2)
    """
    R = 1
    x_dot[0] = args.gamma*1/4 * 1/np.sqrt(np.prod(x)) * ((x[1] / x[0])**(R/2) - (x[0] / x[1])**(R/2)) 
    x_dot[1] = -args.gamma*1/4 * 1/np.sqrt(np.prod(x)) * ((x[1] / x[0])**(R/2) - (x[0] / x[1])**(R/2)) 

    return x_dot

data = []

np.random.seed(42)

for n in range(args.num):
    x = np.array([np.random.random()*1000, np.random.random()*1000], dtype=np.float32)
    x_dot = np.array([np.random.random(),np.random.random()], dtype=np.float32)
    
    time = 0
    dataset = []
    for i in range(args.points):
        dataset.append([time] + [each for each in x] + [each for each in x_dot])

        # Runge Kutta 4th order
        k1i = evolution(x)
        k2i = evolution(x + k1i*args.dt/2)
        k3i = evolution(x + k2i*args.dt/2)
        k4i = evolution(x + k3i*args.dt)

        x_dot = 1/6 * (k1i + 2*k2i + 2*k3i + k4i)
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
    if DIM == 1:
        plt.xlabel("t")
        plt.ylabel("x")
        random_trajectory = data[np.random.randint(0,args.num-1)]
        plt.plot(random_trajectory[:,0], random_trajectory[:,1])
        plt.show()

    if DIM == 2:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        random_trajectory = data[np.random.randint(0,args.num-1)]

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("t")
        ax.plot(random_trajectory[:,1], random_trajectory[:,2], random_trajectory[:,0], label="original data")

        plt.show()

        
    else:
        raise Exception("Plotting is not supported for more dimensions than 1.")
    