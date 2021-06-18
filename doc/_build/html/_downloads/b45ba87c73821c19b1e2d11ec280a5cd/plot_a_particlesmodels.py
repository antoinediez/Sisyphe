"""
Tutorial 01: Particles and models 
============================================

""" 

###################################
# A particle system is an instance of one of the classes defined in the module :mod:`sisyphe.particles`. 
#
# Particles 
#         The basic class :class:`sisyphe.particles.Particles` defines a particle system by the positions. 
#         
# Kinetic particles
#         The class :class:`sisyphe.particles.KineticParticles` defines a particle system by the positions and the velocities.
# 
# Body-oriented particles.
#         The class :class:`sisyphe.particles.BOParticles` defines a particle system in 3D by the positions and the body-orientations which are a rotation matrices in :math:`SO(3)` stored as quaternions. 
# 
# A model is a subclass of a particle class. Several examples are defined in the module :mod:`sisyphe.models`. For example, let us create an instance of the Vicsek model :class:`sisyphe.models.Vicsek` which is a subclass of :class:`sisyphe.particles.KineticParticles`. 
# 
# First, some standard imports...
#

import time 
import torch

###########################################################
# If CUDA is available, the computations will be done on the GPU and on the CPU otherwise. The type of the tensors (simple or double precision) are defined by the type of the initial conditions. Here and throughout the documentation, we work with single precision tensors. 

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

########################################################################
# We take initially :math:`N` particles uniformly scattered in a box of size :math:`L` with uniformly sampled directions of motion. 

N = 10000
L = 100 

pos = L*torch.rand((N,2)).type(dtype)
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))

#########################################################################
# Then we define the interaction radius :math:`R`, the speed of the particles :math:`c` and the drift and diffusion coefficients, respectively :math:`\nu` and :math:`\sigma`. 

R = 5.
c = 1.
nu = 3.
sigma = 1.

############################################
# We take a small discretisation time step. 

dt = .01

##################################################
# Finally, we define an instance of the Vicsek model with these parameters. 

from sisyphe.models import Vicsek

simu = Vicsek(
    pos = pos,
    vel = vel, 
    v = c, 
    sigma = sigma, 
    nu = nu, 
    interaction_radius = R,
    box_size = L,
    dt = dt)

#################################################
# .. note::
#         The boundary conditions are periodic by default, see :ref:`tuto_boundaryconditions`.

######################################################
# So far, nothing has been computed. All the particles are implemented as Python iterators: in order to compute the next time step of the algorithm, we can call the method :meth:`__next__`. This method increments the iteration counter by one and updates all the relevant quantities (positions and velocities) by calling the method :meth:`update() <sisyphe.models.Vicsek.update>` which defines the model. 

print("Current iteration: "+ str(simu.iteration))
simu.__next__()
print("Current iteration: "+ str(simu.iteration))


############################################
# On a longer time interval, we can use the methods in the module :mod:`sisyphe.display`. For instance, let us fix a list of time frames. 

frames = [5., 10., 30., 50., 75., 100]

#############################################
# Using the method :meth:`sisyphe.display.display_kinetic_particles`, the simulation will run until the last time in the list :data:`frames`. The method also displays a scatter plot of the particle system at each of the times specified in the list and finally compute and plot the order parameter. 

from sisyphe.display import display_kinetic_particles

s = time.time()
it, op = display_kinetic_particles(simu, frames, order=True)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')




    
    
    
    
    