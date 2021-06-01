"""
Bands
============================================

The classical Vicsek model in a square periodic domain `is known <https://arxiv.org/abs/0712.2062>`_ to produce band-like structures in a very dilute regime. These structures also appears in a mean-field regime. To showcase the efficiency of the SiSyPHE library, we simulate a mean-field :class:`Vicsek <sisyphe.models.Vicsek>` model with the target :meth:`max_kappa() <sisyphe.particles.KineticParticles.max_kappa>` and :math:`10^6` particles. 

""" 

###################################
# First of all, some standard imports. 

import os
import sys
import time
import math
import torch
import numpy as np 
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


#######################################
# Set the parameters and create an instance of the Vicsek model. 

import sisyphe.models as models

N = 1000000
L = 1.
dt = .01

nu = 3
sigma = 1.
kappa = nu/sigma

R = .01
c = .1

pos = L*torch.rand((N,2)).type(dtype)
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))

simu=models.Vicsek(pos=pos,vel=vel,
             v=c,
             sigma=sigma,nu=nu,
             interaction_radius=R,
             box_size=L,
             boundary_conditions='periodic',
             variant = {"name" : "max_kappa", "parameters" : {"kappa_max" : 10.}},
             options = {},
             numerical_scheme='projection',
             dt=dt,
             block_sparse_reduction=True)


##############################################
# Check that we are in a mean field regime... 

Nneigh = simu.number_of_neighbours()

print("The most isolated particle has " + str(Nneigh.min().item()) + " neighbours.")
print("The least isolated particle has " + str(Nneigh.max().item()) + " neighbours.")


##########################################
# Set the block sparse parameters to their optimal value. 

fastest, nb_cells, average_simu_time, simulation_time = simu.best_blocksparse_parameters(40,100)

plt.plot(nb_cells,average_simu_time)
plt.show()

###########################################
# Create the function which compute the center of mass of the system (on the torus).

def center_of_mass(particles):
    cos_pos = torch.cos((2*math.pi / L) * particles.pos)
    sin_pos = torch.sin((2*math.pi / L) * particles.pos)
    average_cos = cos_pos.sum(0)
    average_sin = sin_pos.sum(0)
    center = torch.atan2(average_sin, average_cos)
    center = (L / (2*math.pi)) * torch.remainder(center, 2*math.pi)
    return center


############################################
# Let us save the positions and velocities of 100k particles and the center of mass of the system during 300 units of time. 

from sisyphe.display import save

frames = [50., 100., 300.]

s = time.time()
data = save(simu,frames,["pos", "vel"],[center_of_mass], Nsaved=100000, save_file=False)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')


#############################################
# At the end of the simulation, we plot the particles and the evolution of the center of mass. 

# sphinx_gallery_thumbnail_number = 2
f = plt.figure(0, figsize=(12, 12))
for frame in range(len(data["frames"])):
    x = data["pos"][frame][:,0].cpu()
    y = data["pos"][frame][:,1].cpu()
    u = data["vel"][frame][:,0].cpu()
    v = data["vel"][frame][:,1].cpu()
    ax = f.add_subplot(2,2,frame+1)
    plt.quiver(x,y,u,v)
    ax.set_xlim(xmin=0, xmax=simu.L[0].cpu())
    ax.set_ylim(ymin=0, ymax=simu.L[1].cpu())
    ax.set_title("time="+str(data["frames"][frame]))

center = data["center_of_mass"]

center_x = []
center_y = []

for c in center:
    center_x.append(c[0])
    center_y.append(c[1])

f = plt.figure(1)
plt.plot(data["time"],center_x)
plt.ylabel("x-coordinate of the center of mass")
plt.xlabel("time")


f = plt.figure(2)
plt.plot(data["time"],center_y)
plt.ylabel("y-coordinate of the center of mass")
plt.xlabel("time")
plt.show()

###############################################################
# We are still in a mean-field regime. 

Nneigh = simu.number_of_neighbours()

print("The most isolated particle has " + str(Nneigh.min().item()) + " neighbours.")
print("The least isolated particle has " + str(Nneigh.max().item()) + " neighbours.")



