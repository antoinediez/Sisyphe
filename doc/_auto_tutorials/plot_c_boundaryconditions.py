"""
.. _tuto_boundaryconditions:

Tutorial 03: Boundary conditions
=======================================

The particle systems are defined in a rectangular box whose dimensions are specified by the attribute :attr:`L <sisyphe.Particles.particles.L>`. 

The boundary conditions are specified by the attribute :attr:`bc <sisyphe.Particles.particles.bc>` which can be one of the following. 

* A list of size :math:`d` containing for each dimension either 0 (periodic) or 1 (wall with reflecting boundary conditions).

* The string ``"open"`` : no boundary conditions.

* The string ``"periodic"`` : periodic boundary conditions.

* The string ``"spherical"`` : reflecting boundary conditions on the sphere of diameter :math:`L` enclosed in the square domain :math:`[0,L]^d`. 

""" 

###########################################################
# For instance, let us simulate the Vicsek model in an elongated rectangular domain :math:`[0,L_x]\times[0,L_y]` with periodic boundary conditions in the :math:`x`-dimension and reflecting boundary conditions in the :math:`y`-dimension. 
#
# First, some standard imports...
#

import time 
import torch
from sisyphe.models import Vicsek
from sisyphe.display import display_kinetic_particles

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

################################################
# The parameters of the model 

N = 100000

R = .01
c = .1
nu = 3.
sigma = 1.

dt = .01

variant = {"name" : "max_kappa", "parameters" : {"kappa_max" : 10.}}

################################################
# The spatial domain, the boundary conditions and the initial conditions... 
# 

Lx = 3.
Ly = 1./3.
L = [Lx, Ly]
bc = [0,1]

pos = torch.rand((N,2)).type(dtype)
pos[:,0] = L[0]*pos[:,0]
pos[:,1] = L[1]*pos[:,1]
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))

simu = Vicsek(
    pos = pos.detach().clone(),
    vel = vel.detach().clone(), 
    v = c, 
    sigma = sigma, 
    nu = nu, 
    interaction_radius = R,
    box_size = L,
    boundary_conditions=bc,
    dt = dt,
    variant = variant,
    block_sparse_reduction = True,
    number_of_cells = 100**2)


######################################################
# Finally run the simulation over 300 units of time.... 

# sphinx_gallery_thumbnail_number = 15

frames = [0., 2., 5., 10., 30., 42., 71., 100, 123, 141, 182, 203, 256, 272, 300]

s = time.time()
it, op = display_kinetic_particles(simu, frames, order=True, figsize=(8,3))
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')

##################################################
# The simulation produces small clusters moving from left to right or from right to left. Each "step" in the order parameter corresponds to a collision between two clusters moving in opposite directions. 

