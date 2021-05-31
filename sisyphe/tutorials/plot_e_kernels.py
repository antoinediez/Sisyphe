"""
.. _tuto_averages:

Tutorial 05: Kernels and averages
============================================

Simulating swarming models requires expensive mean-field convolution operations of the form: 

.. math::
    
    J^i = \\frac{1}{N}\sum_{j=1}^N K(|X^j-X^i|) U^j,
    
for :math:`1\leq i\leq N`, where :math:`(X^i)_{1\leq i \leq N}` are the positions of the particles, :math:`(U^j)_{1\leq j\leq N}` are given vectors and :math:`K` is an **observation kernel**. Typically, :math:`K(|X^i-X^j|)` is equal to 1 if :math:`X^i` and :math:`X^j` are at distance smaller than a fixed interaction distance and 0 otherwise. Other kernels are defined in the module :mod:`sisyphe.kernels`. Below, we show a simple application case. 

""" 

###################################
# Linear local averages
# ---------------------------
#
# First, some standard imports...
#

import time 
import math
import torch
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

##################################################
# Let the :math:`N` particles be uniformly scattered in a box of size :math:`L` with interaction radius  :math:`R`.
# 

N = 100000
L = 1. 
R = .15

pos = L*torch.rand((N,2)).type(dtype)


###############################
# We can also assume that the particles have a bounded cone of vision around an axis (defined by a unit vector). The default behaviour is a full vision angle equal to :math:`2\pi` in which case the axis is a :data:`None` object. Here we take a cone of vision with angle :math:`\pi/2` around an axis which is sampled uniformly. For the :class:`sisyphe.particles.KineticParticles`, the default axis is the velocity. 

angle = math.pi/2
axis = torch.randn(N,2).type(dtype)
axis = axis/torch.norm(axis,dim=1).reshape((N,1))

#################################
# Let us create an instance of a particle system with these parameters. 

from sisyphe.particles import Particles 

particles = Particles(
    pos = pos,
    interaction_radius = R,
    box_size = L,
    vision_angle = angle,
    axis = axis)

####################################
# .. note::
#         By default, the system and the operations below are defined with periodic boundary conditions. 

##################################
# As a simple application, we can compute the number of neighbours of each particle and print the number of neighbours of the first particle. This operation is already implemented in the method :func:`number_of_neighbours() <sisyphe.particles.Particles.number_of_neighbours>`. It simply corresponds to the average: 
#
# .. math::
#         
#         N^i_\mathrm{neigh} = \sum_{j=1}^N K(|X^j-X^i|).

Nneigh = particles.number_of_neighbours()

Nneigh0 = int(Nneigh[0].item())

print("The first particle sees " + str(Nneigh0) + " other particles.")

#######################################
# For custom objects, the mean-field average can be computed using the method :func:`linear_local_average() <sisyphe.particles.Particles.linear_local_average>`. As an example, let us compute the center of mass of the neighbours of each particle. First we define the quantity :math:`U` that we want to average. Here it is simply the positions of the particles. 

U = particles.pos

######################################
# Then we compute the mean field average which is the standard convolution over the :math:`N` particles. To get the center of mass, we simply renormalise this average by the actual number of neighbours. 

mean_field_average, = particles.linear_local_average(U)
center_of_mass = N/Nneigh.reshape((N,1)) * mean_field_average

#############################################
# In the method :func:`linear_local_average() <sisyphe.particles.Particles.linear_local_average>`, the default observation kernel is a :class:`LazyTensor` of size :math:`(N,N)` whose :math:`(i,j)` component is equal to 1 when particle :math:`j` belongs to the cone of vision of particle :math:`i` and 0 otherwise. To retrieve the indexes of the particles which belong to the cone of vision of the first particle, we can use the `K-nearest-neighbours reduction <https://www.kernel-operations.io/keops/_auto_tutorials/knn/plot_knn_mnist.html#sphx-glr-auto-tutorials-knn-plot-knn-mnist-py>`_ provided by the `KeOps <https://www.kernel-operations.io/keops/index.html>`_ library. 

from sisyphe.kernels import lazy_interaction_kernel

interaction_kernel = lazy_interaction_kernel(
    particles.pos, 
    particles.pos, 
    particles.R,
    particles.L,
    boundary_conditions = particles.bc,
    vision_angle = particles.angle,
    axis = particles.axis)

K_ij = 1. - interaction_kernel 

neigh0 = K_ij.argKmin(Nneigh0, dim=1)[0]

print("The indexes of the neighbours of the first particles are: ")
print(neigh0)

####################################################
# Finally, a fancy display of what we have computed. We plot the full particle system in black, the first particle in orange, its neighbours in blue and the center of mass of the neighbours in red. 

xall = particles.pos[:,0].cpu().numpy()
yall = particles.pos[:,1].cpu().numpy()

x = particles.pos[neigh0,0].cpu().numpy()
y = particles.pos[neigh0,1].cpu().numpy()

x0 = particles.pos[0,0].item()
y0 = particles.pos[0,1].item()

xc = center_of_mass[0,0].item()
yc = center_of_mass[0,1].item()


fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(xall, yall, s=.003, c='black')
ax.scatter(x, y, s=.3)
ax.scatter(x0, y0, s=24)
ax.scatter(xc, yc, s=24, c='red')
ax.axis([0, L, 0, L])
ax.set_aspect("equal")

#######################################################
# Nonlinear averages
# ---------------------------------
#
# In some cases, we need to compute a **nonlinear average** of the form 
#
# .. math::
#         
#         J^i = \frac{1}{N}\sum_{j=1}^N K(|X^j-X^i|) b(U^i,V^j),
#
# where :math:`(U^i)_{1\leq i \leq N}` and :math:`(V^j)_{1\leq j \leq N}` are given vectors and :math:`b` is a given function. When the **binary formula** :math:`b` can be written as a :class:`LazyTensor`, this can be computed with the method :func:`nonlinear_local_average() <sisyphe.particles.Particles.nonlinear_local_average>`. 
#
# For instance, let us compute the local mean square distance: 
#
# .. math::
#         
#         J^i = \frac{\sum_{j=1}^N K(|X^j-X^i|) |X^j-X^i|^2}{\sum_{j=1}^N K(|X^j-X^i|)}.
#
# In this case, we can use the function :func:`sisyphe.kernels.lazy_xy_matrix` to define a custom binary formula. Given two vectors :math:`X=(X^i)_{1\leq i\leq M}` and :math:`Y = (Y^j)_{1\leq j\leq N}`, respectively of sizes :math:`(M,d)` and :math:`(N,d)`, the :math:`XY` matrix is a :math:`(M,N,d)` LazyTensor whose :math:`(i,j,:)` component is the vector :math:`Y^j-X^i`. 
#

from sisyphe.kernels import lazy_xy_matrix 

def b(x,y): 
    K_ij = lazy_xy_matrix(x,y,particles.L)
    return (K_ij ** 2).sum(-1)

x = particles.pos
y = particles.pos
mean_square_dist = N/Nneigh.reshape((N,1)) * particles.nonlinear_local_average(b,x,y)

#################################################
# Since the particles are uniformly scattered in the box, the theoretical value is 
# 
# .. math::
#         
#         MSD_0 = \frac{\int_0^R \int_0^{\pi/2} r^3 \mathrm{d}r\mathrm{d}\theta}{\int_0^R \int_0^{\pi/2} r \mathrm{d}r\mathrm{d}\theta} = \frac{R^2}{2}
# 

print("Theoretical value: " + str(R**2/2))
print("Experimental value: " + str(mean_square_dist[0].item()))






    
    
    
    
    