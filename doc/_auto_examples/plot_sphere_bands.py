"""
Boundary clusters in a disk
============================================

A classical mean-field Vicsek model in a bounded disk domain.  

""" 

###################################
# First of all, some standard imports. 

import os
import sys
import time
import torch
import numpy as np 
from matplotlib import pyplot as plt
from sisyphe.display import display_kinetic_particles


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


#######################################
# Set the parameters and create an instance of the Vicsek model. 

import sisyphe.models as models

N = 1000000
L = 10.
dt = .01

nu = 3
sigma = 1.
kappa = nu/sigma

R = .1
c = 1.

center = torch.tensor([L/2,L/2]).type(dtype).reshape((1,2))
radius = L/2
pos = L*torch.rand((N,2)).type(dtype)
out = ((pos-center)**2).sum(1) > radius**2
while out.sum()>0:
    pos[out,:] = L*torch.rand((out.sum(),2)).type(dtype)
    out = ((pos-center)**2).sum(1) > radius**2
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))

simu=models.Vicsek(pos=pos,vel=vel,
             v=c,
             sigma=sigma,nu=nu,
             interaction_radius=R,
             box_size=L,
             boundary_conditions='spherical',
             variant = {"name" : "max_kappa", "parameters" : {"kappa_max" : 10.}},
             options = {},
             numerical_scheme='projection',
             dt=dt,
             block_sparse_reduction=True)


##########################################
# Set the block sparse parameters to their optimal value. 

fastest, nb_cells, average_simu_time, simulation_time = simu.best_blocksparse_parameters(40,100)

plt.plot(nb_cells,average_simu_time)
plt.show()


##########################################
# Run the simulation and plot the particles. 

# sphinx_gallery_thumbnail_number = -1

frames = [0, 10, 40, 70, 100, 150, 200, 250, 300]

s = time.time()
it, op = display_kinetic_particles(simu, frames, N_dispmax=100000)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')



