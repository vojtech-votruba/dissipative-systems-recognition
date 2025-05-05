import argparse
import numpy as np
import os
from matplotlib import pyplot as plt
import scienceplots

"""
    Code for generating linear diffusion trajectories in the spirit of gradient dynamics.
    The method of numerical integration we leverage is the standard 4th order Runge--Kutta.
"""

parser = argparse.ArgumentParser(prog='simulate_trajectory.py',
                                description='A short script for generating gradient dynamics data used in a machine learning project.',)

parser.add_argument("--num" , default=1024, type=int, help="number of trajectories that we want to simulate")
parser.add_argument("--points", default=64, type=int, help="number of points in time for each trajectory")
parser.add_argument("--dt", default=0.03, type=float, help="size of the time step used in the simulation")
parser.add_argument("--plot", default=True, type=bool, help="plot the results")
args = parser.parse_args()

np.random.seed(42)

X_GRID = np.linspace(0,1,64)
h = 1/64
D = 3e-3


def polynomial_init(N, x):
    p = 0
    for q in range(N):
        a_n = np.random.uniform(-1,1)
        p += a_n*x**q
    
    f = (1-x)*x*p
    f += abs(np.min(f))

    return f

def discont_init(N, x):
    f = np.zeros_like(x)
    for i in range(N):
        h = np.random.uniform(0.1,1)
        width = np.random.uniform(1/(4*N), 1/(2*N))
        centre = np.random.uniform(1/N, 1 - 1/N)
        mask = (x >= centre - width/2) & (x < centre + width/2)
        f[mask] += h

    return f

def exp_peak_init(dist, x):
    return np.exp(-100*(x-(0.5-dist))**2) + np.exp(-100*(x-(0.5+dist))**2)

def fourier_init(N, x, p=1.5):
    f = np.zeros_like(x)
    constant_term = np.random.uniform(-0.5, 0.5)
    f += constant_term

    for n in range(1, N + 1):
        decay = 1 / (n ** p)
        a_n = np.random.normal(0, decay)
        b_n = np.random.normal(0, decay)
        f += b_n * np.sin(2 * np.pi * n * x) + a_n * np.cos(2 * np.pi * n * x)

    return f

def evolution(c):
    c_dot = np.zeros_like(c)
    c_dot[1:-1] = D * (c[2:] - 2*c[1:-1] + c[:-2]) / (h**2)
    
    c_dot[0] = D * (c[-1] -2*c[0] + c[1])/ h**2
    c_dot[-1] = D * (c[-2] -2*c[-1] + c[0])/ h**2

    return c_dot

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

for n in range(args.num):
    c = fourier_init(np.random.randint(1, 15), X_GRID)
    #c = polynomial_init(np.random.randint(1, 5), X_GRID)
    #c = discont_init(np.random.randint(1, 5), X_GRID)
    #c = exp_peak_init(np.random.uniform(0.1,0.35), X_GRID)

    c = c/np.linalg.norm(c)

    time = 0
    dataset = []
    for i in range(args.points):
        dataset.append([time] + [each for each in c])

        c_dot = rk4(evolution, c, args.dt)
        c += c_dot * args.dt
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

print("\nDone! Trajectories saved into ./data/dataset.txt")

if args.plot:
    plt.style.use(['science','ieee'])

    fig, ax = plt.subplots()
    random_trajectory = data[np.random.randint(0,args.num-1)]

    for seq in range(4):
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$c$")
        
        c_at_t = random_trajectory[int((len(random_trajectory)-1) * seq/3)][1:]
        time = random_trajectory[int((len(random_trajectory)-1) * seq/3)][0]

        ax.plot(X_GRID, c_at_t, label=f"$t = {time:.1e}$")
        ax.legend()
    
    fig.savefig("results/example.pdf")
    plt.show()
        