"""
.. _examplemill:

Mills
============================================

Examples of milling behaviours. 

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
import sisyphe.models as models
from sisyphe.display import display_kinetic_particles

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


#######################################
# Milling in the D'Orsogna et al. model
# -----------------------------------------
#
# Let us create an instance of the attraction-repulsion model introduced in
# 
# M. R. D’Orsogna, Y. L. Chuang, A. L. Bertozzi, L. S. Chayes, Self-Propelled Particles with Soft-Core Interactions: Patterns, Stability, and Collapse, *Phys. Rev. Lett.*, Vol. 96, No. 10 (2006).
#
# The particle system satisfies the ODE:
#
# .. math::  
#        
#         \frac{\mathrm{d}X^i_t}{\mathrm{d}t} = V^i_t
#        
# .. math::
#        
#          \frac{\mathrm{d}V^i_t}{\mathrm{d}t} = (\alpha-\beta|V^i_t|^2)V^i_t - \frac{m}{N}\nabla_{x^i}\sum_{j\ne i} U(|X^i_t-X^j_t|)
#
# where :math:`U` is the Morse potential 
#
# .. math::
#
#        U(r) := -C_a\mathrm{e}^{-r/\ell_a}+C_r\mathrm{e}^{-r/\ell_r}
#
#


N = 10000
mass = 1000.
L = 10. 

Ca = .5
la = 2.
Cr = 1.
lr = .5

alpha = 1.6
beta = .5
v0 = math.sqrt(alpha/beta)

pos = L*torch.rand((N,2)).type(dtype)
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))
vel = v0*vel

dt = .01

simu = models.AttractionRepulsion(pos=pos,
                 vel=vel,
                 interaction_radius=math.sqrt(mass),
                 box_size=L,
                 propulsion = alpha,
                 friction = beta,                        
                 Ca = Ca,
                 la = la,
                 Cr = Cr,
                 lr = lr,                        
                 dt=dt,
                 p=1,                        
                 isaverage=True)

##########################################
# Run the simulation over 100 units of time and plot 10 frames. The ODE system is solved using the Runge-Kutta 4 numerical scheme. 
#

frames = [0,1,2,3,4,5,10,40,70,100]

s = time.time()
it, op = display_kinetic_particles(simu,frames)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')



##########################################
# Milling in the Vicsek model 
# ----------------------------------------
# 
# Let us create an instance of the Asynchronuos Vicsek model with a bounded cone of vision and a bounded angular velocity, as introduced in: 
# 
# A. Costanzo, C. K. Hemelrijk, Spontaneous emergence of milling (vortex state) in a Vicsek-like model, *J. Phys. D: Appl. Phys.*, 51, 134004
#
#

N = 10000
R = 1.
L = 20.

nu = 1
sigma = .02
kappa = nu/sigma

c = .175
angvel_max = .175/nu

pos = L*torch.rand((N,2)).type(dtype)
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))

dt = .01

################################################
# We add an option to the target

target = {"name" : "normalised", "parameters" : {}}
option = {"bounded_angular_velocity" : {"angvel_max" : angvel_max, "dt" : 1./nu}}

simu=models.AsynchronousVicsek(pos=pos,vel=vel,
                 v=c,
                 jump_rate=nu,kappa=kappa,
                 interaction_radius=R,
                 box_size=L,
                 vision_angle=math.pi, axis = None,
                 boundary_conditions='periodic',
                 variant=target,
                 options=option,
                 sampling_method='projected_normal')


##########################################
# Run the simulation over 200 units of time and plot 10 frames. 
#

# sphinx_gallery_thumbnail_number = -1

frames = [0, 10, 30, 50, 75, 100, 125, 150, 175, 200]

s = time.time()
it, op = display_kinetic_particles(simu,frames)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')



