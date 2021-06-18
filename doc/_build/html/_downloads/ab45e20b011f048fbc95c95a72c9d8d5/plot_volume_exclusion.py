"""
Volume exclusion
============================================

""" 

#####################################
# This model is introduced in
#
# S. Motsch, D. Peurichard, From short-range repulsion to Hele-Shaw problem in a model of tumor growth, *J. Math. Biology*, Vol. 76, No. 1, 2017.


###################################
# First of all, some standard imports. 

import os
import sys
import time
import math
import torch
import numpy as np 
from matplotlib import pyplot as plt
import sisyphe.models as models
from sisyphe.display import scatter_particles


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


#######################################
# Repulsion force
# -------------------------------------------
#
# Each particle is a disk with a (fixed) random radius. The particles repel each other when they overlap. The force exerted by a particle located at :math:`x_j` with radius :math:`R_j` on a particle located at :math:`x_i` with radius :math:`R_i` is
#
# .. math::
#
#     F = -\frac{\alpha}{R_i} \nabla_{x_i} U\left(\frac{|x_i - x_j|^2}{(R_i + R_j)^2}\right),
#
# where the potential is
#
# .. math::
#
#     U(s) = -\log(s) + s - 1\,\,\text{for}\,\, s<1 \,\,\text{and}\,\, U(s) = 0\,\, \text{for}\,\, s>1.
#
# Initially, the particles are clustered in a small region with a strong overlapping. 
#


N = 10000
rmin = .1
rmax = 1.
R = (rmax-rmin)*torch.rand(N).type(dtype)+rmin
L = 100.
D0 = 20.
pos = (D0*torch.rand((N,2)).type(dtype)-D0/2)+torch.tensor([L/2,L/2]).type(dtype)

dt = .1

simu = models.VolumeExclusion(pos=pos,
                 interaction_radius=R,
                 box_size=L,
                 alpha=2.5,
                 division_rate=0., 
                 death_rate=0.,                    
                 dt=dt)

##########################################
# Run the simulation over 200 units of time using an adaptive time-step which ensures that the energy :attr:`E` of the system decreases.
#

# sphinx_gallery_thumbnail_number = 13

frames = [0,1,2,3,4,5,10,30,50,75,100,150,200]

s = time.time()
scatter_particles(simu,frames)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')

#############################################
# Note this funny behaviour: the particles are clustered by size! 

#######################################
# Repulsion force, random births and random deaths
# ---------------------------------------------------------------------
#
# Same system but this time, particles die at a constant rate and give birth to new particles at the same rate. A new particle is added next to its parent and has the same radius. 
#

N = 10000
rmin = .1
rmax = 1.
R = (rmax-rmin)*torch.rand(N).type(dtype)+rmin
L = 100.
D0 = 20.
pos = (D0*torch.rand((N,2)).type(dtype)-D0/2)+torch.tensor([L/2,L/2]).type(dtype)

dt = .1

simu = models.VolumeExclusion(pos=pos,
                 interaction_radius=R,
                 box_size=L,
                 alpha=2.5,
                 division_rate=.3, 
                 death_rate=.3,                    
                 dt=dt,
                 Nmax = 20000)

##########################################
# Run the simulation over 200 units of time using an adaptive time-step which ensures that the energy :attr:`E <sisyphe.models.VolumeExclusion.E>` of the system decreases.
#

frames = [0,1,2,3,4,5,10,30,50,75,100,150,200]

s = time.time()
scatter_particles(simu,frames)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')

