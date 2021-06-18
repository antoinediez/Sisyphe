"""
Tutorial 02: Targets and options
=======================================

Many useful **local averages** (see :ref:`tuto_averages`) are pre-defined and may be directly used in the simulation of new or classical models. This tutorial showcases the basic usage of the **target methods** to simulate the variants of the Vicsek model. 

""" 

#########################################
# An example: variants of the Vicsek model
# ------------------------------------------
#
# In its most abstract form, the Vicsek model reads: 
#
# .. math:: 
#         
#         \mathrm{d}X^i_t = c_0 V^i_t \mathrm{d} t
#         
# .. math::
#         
#         \mathrm{d}V^i_t = \sigma\mathsf{P}(V^i_t)\circ ( J^i_t \mathrm{d}t + \mathrm{d} B^i_t),
# 
# where :math:`c_0` is the speed, :math:`\mathsf{P}(v)` is the orthogonal projection on the orthogonal plane to :math:`v` and :math:`\sigma` is the diffusion coefficient. The drift coefficient :math:`J^i_t` is called a **target**. In synchronous and asynchronous Vicsek models, the target is the center of the sampling distribution. A comparison of classical targets is shown below.


##########################################
# First, some standard imports...
#

import time 
import torch
import pprint
from matplotlib import pyplot as plt
from sisyphe.models import Vicsek
from sisyphe.display import display_kinetic_particles

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

############################################
# The classical non-normalised Vicsek model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The target is: 
# 
# .. math:: 
#         
#         J^i_t = \kappa\frac{\sum_{j=1}^N K(|X^j_t-X^i_t|)V^j_t}{|\sum_{j=1}^N K(|X^j_t-X^i_t|)V^j_t|}.
# 
# The parameters of the model... 
#


N = 100000
L = 100 

pos = L*torch.rand((N,2)).type(dtype)
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))

R = 3.
c = 3.
nu = 5.
sigma = 1.

dt = .01

################################################
# The choice of the target is implemented in the keyword argument ``variant``. For the classical normalised target, it is given by the following dictionary.
# 

variant = {"name" : "normalised", "parameters" : {}}

simu = Vicsek(
    pos = pos.detach().clone(),
    vel = vel.detach().clone(), 
    v = c, 
    sigma = sigma, 
    nu = nu, 
    interaction_radius = R,
    box_size = L,
    dt = dt,
    variant = variant,
    block_sparse_reduction = True,
    number_of_cells = 40**2)


######################################################
# Finally run the simulation over 300 units of time.... 

frames = [0., 5., 10., 30., 42., 71., 100, 124, 161, 206, 257, 300]

s = time.time()
it, op = display_kinetic_particles(simu, frames, order=True)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')

####################################################
# Plot the histogram of the angles of the directions of motion. 

angle = torch.atan2(simu.vel[:,1],simu.vel[:,0])
angle = angle.cpu().numpy()
h = plt.hist(angle, bins=1000)
plt.xlabel("angle")
plt.show()

##########################################
# After an initial clustering phase, the system self-organizes into a uniform flock. 

####################################
# Non-normalised Vicsek model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The target is: 
# 
# .. math:: 
#         
#         J^i_t = \frac{\frac{1}{N}\sum_{j=1}^N K(|X^j_t-X^i_t|)V^j_t}{\frac{1}{\kappa}+\frac{1}{\kappa_0}|\frac{1}{N}\sum_{j=1}^N K(|X^j_t-X^i_t|)V^j_t|}.
# 
# Define the corresponding dictionary...

kappa_0 = 15.

variant = {"name" : "max_kappa", "parameters" : {"kappa_max" : kappa_0}}

simu = Vicsek(
    pos = pos.detach().clone(),
    vel = vel.detach().clone(), 
    v = c, 
    sigma = sigma, 
    nu = nu, 
    interaction_radius = R,
    box_size = L,
    dt = dt,
    variant = variant,
    block_sparse_reduction = True,
    number_of_cells = 40**2)


######################################################
# Finally run the simulation over 300 units of time.... 

frames = [0., 5., 10., 30., 42., 71., 100, 124, 161, 206, 257, 300]

s = time.time()
it, op = display_kinetic_particles(simu, frames, order=True)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')

####################################################
# Plot the histogram of the angles of the directions of motion. 

angle = torch.atan2(simu.vel[:,1],simu.vel[:,0])
angle = angle.cpu().numpy()
h = plt.hist(angle, bins=1000)
plt.xlabel("angle")
plt.show()

#############################################################
# The system self-organizes into a strongly clustered flock with band-like structures. 


##########################
# Nematic Vicsek model 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The target is: 
# 
# .. math:: 
#         
#         J^i_t = \kappa (V^i_t\cdot \overline{\Omega}^i_t)\overline{\Omega}^i_t,
#
# where :math:`\overline{\Omega}^i_t` is any unit eigenvector associated to the maximal eigenvalue of the average Q-tensor:
#
# .. math::
#         
#         Q^i_t = \frac{1}{N}\sum_{j=1}^N K(|X^j_t-X^i_t|){\left(V^j_t\otimes V^j_t - \frac{1}{d} I_d\right)}.
# 
# Define the corresponding dictionary...

variant = {"name" : "nematic", "parameters" : {}}

simu = Vicsek(
    pos = pos.detach().clone(),
    vel = vel.detach().clone(), 
    v = c, 
    sigma = sigma, 
    nu = nu, 
    interaction_radius = R,
    box_size = L,
    dt = dt,
    variant = variant,
    block_sparse_reduction = True, 
    number_of_cells = 40**2)

######################################################
# Finally run the simulation over 100 units of time... The color code indicates the angle of the direction of motion between :math:`-\pi` and :math:`\pi`. 

frames = [0., 5., 10., 30., 42., 71., 100]

s = time.time()
it, op = display_kinetic_particles(simu, frames, order=True, color=True)
e = time.time()

##################################################
# Print the total simulation time and the average time per iteration. 

print('Total time: '+str(e-s)+' seconds')
print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')

####################################################
# Plot the histogram of the angles of the directions of motion. 

# sphinx_gallery_thumbnail_number = -1

angle = torch.atan2(simu.vel[:,1],simu.vel[:,0])
angle = angle.cpu().numpy()
h = plt.hist(angle, bins=1000)
plt.xlabel("angle")
plt.show()

###################################################
# There are two modes separated by an angle :math:`\pi` which indicates that two groups of equal size are moving in opposite direction. This a *nematic flock*. 


#####################################################
# The target dictionary 
# --------------------------------
#
# Several other targets are implemented. The complete list of available targets can be found in the **dictionary of targets** :attr:`target_method <sisyphe.particles.Particles.target_method>`.

pprint.pprint(simu.target_method)

#####################################################
# Custom targets can be added to the dictionary of targets using the method :meth:`add_target_method() <sisyphe.particles.Particles.add_target_method>`. Then a target can be readily used in a simulation using the method :meth:`compute_target() <sisyphe.particles.Particles.compute_target>`. 


################################################
# Options
# ---------------
#
# Customized targets can also be defined by applying an **option** which modifies an existing target. See :ref:`examplemill`.  





