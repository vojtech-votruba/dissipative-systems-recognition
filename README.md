# About
In this work, we design and implement neural networks specialized in modeling dissipative systems. For this purpose, we leverage a theory called generalized gradient dynamics that has the unique property of generating equations that automatically satisfy the second law of thermodynamics. Neural networks then serve as a powerful tool for reconstructing non-trivial physical models directly from experimental or simulated data. After introducing core concepts and taking inspiration from existing approaches, we introduce our method, which we successfully demonstrate by recognizing the dynamics of several well-known systems, e.g., chemical reactions. While the proposed approach can be used as an alternative to traditional numerical methods, the main future goal is to connect it with non-dissipative neural networks and apply it to the modeling of any physical, chemical, or biological systems where the exact evolution equations are unknown.

# How to run
Three examples on which we illustrate the recongition of dissipative systems using generalized gradient dynamics are located in the folders:
```
overdamped_particle/
chemical_reactions/
fickian_diffusion/
```
Each folder contains a `main.py` file, through which one can train or use the deep learning model. Before starting the training, one must run the `simuluate_trajectory.py` script which simulates an artifical dataset and stores it into `/data/` folder.

Chemical reactions further contain two additional files corresponding to training the model with either prescribed entropy or prescribed dissipation potential.

# Requirements
The requirements are located in the `requirements.txt` file. We recommend using a virtual environment, specifically with the conda package manager. To get the requirements with conda installed, you can use
```
conda install -y requirements.txt
``` 