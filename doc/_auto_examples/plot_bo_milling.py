"""
Body-oriented mill
============================================

""" 

########################################
# This model is introduced in 
#
# P. Degond, A. Diez, M. Na, Bulk topological states in a new collective dynamics model,  arXiv:2101.10864, 2021 
#

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
from sisyphe.display import save

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


#######################################
# Body-oriented particles with initial perpendicular twist
# ---------------------------------------------------------------
#
# The system is composed of body-oriented particles which are initially uniformly scattered in a periodic box but their body-orientations are "twisted". The body orientation of a particle at position :math:`(x,y,z)` is initially: 
#
# .. math:: 
#    
#     \left(\begin{array}{ccc} 1 & 0 & 0 \\ 0 & \cos(2\pi z) & -\sin(2\pi z) \\ 0 & \sin(2\pi z) & \cos(2\pi z)\end{array}\right).
#
#


from sisyphe.initial import cyclotron_twist_z

N = 1500000
L = 1
R = .025
nu = 40
c = 1
kappa = 10

pos, bo = cyclotron_twist_z(N,L,1,kappa,dtype)

simu = models.BOAsynchronousVicsek(pos=pos,bo=bo,
                 v=c,
                 jump_rate=nu,kappa=kappa,
                 interaction_radius=R,
                 box_size=L,
                 boundary_conditions='periodic',
                 variant = {"name" : "normalised", "parameters" : {}},
                 options = {},
                 sampling_method='vonmises',
                 block_sparse_reduction=True,
                 number_of_cells=15**3)

#################################################
# Run the simulation over 5 units of time and save the azimuthal angle of the mean direction of motion defined by: 
#
# .. math::
#
#     \varphi = \mathrm{arg}(\Omega^1+i\Omega^2) \in [0,2\pi],
#
# where :math:`\Omega = (\Omega^1,\Omega^2,\Omega^3)` is the mean direction of motion of the particles with velocities :math:`(\Omega_i)_{1\leq i\leq N}` : 
#
# .. math::
#
#     \Omega := \frac{\sum_{i=1}^N \Omega_i}{|\sum_{i=1}^N \Omega_i|}
#

frames = [5.]

s = time.time()
data = save(simu,frames,[],["phi"],save_file=False)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 


print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')


#########################################################
# Plot the azimuthal angle :math:`\varphi`. 

plt.plot(data["time"],data["phi"])
plt.xlabel("time")
plt.ylabel("Azimuthal angle")

