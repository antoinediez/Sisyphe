"""
.. _tutobsr:

Tutorial 04: Block-sparse reduction 
============================================

In many cases, the interaction radius :math:`R` is much smaller than the size of the domain. Consequently, the sums in the local averages (see :ref:`tuto_averages`) contain only a small fraction of non zero terms. To gain in efficiency, we can follow the classical strategy:

* Subdivide the domain into a fixed number of cells of size at least :math:`R`.
* For a particle in a given cell, only look at the contiguous cells to compute the local averages. In dimension :math:`d`, there are :math:`3^d` contiguous cells (including the cell itself). 

A practical implementation is called the *Verlet list method*. However, the implementation below is different than the classical one. It is adapted from the `block-sparse reduction method <https://www.kernel-operations.io/keops/_auto_examples/pytorch/plot_grid_cluster_pytorch.html>`_ implemented in the `KeOps <https://www.kernel-operations.io/keops/index.html>`_ library. 

We illustrate the gain in efficency for the Vicsek model. 

.. note::
    The method is sub-optimal for moderate numbers of particles. As a rule of thumb, the block-sparse reduction method becomes useful for systems with at least :math:`10^4` particles. 

""" 

###################################
# Set up and benchmarks
# ---------------------------
#
# First, some standard imports...
#

import copy
import time 
import torch
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

##################################################
# Let the :math:`N` particles be uniformly scattered in a box of size :math:`L` with interaction radius  :math:`R` and uniformly sampled velocities. 
# 

from sisyphe.models import Vicsek

N = 100000
L = 100.  
R = 1.

pos = L*torch.rand((N,2)).type(dtype)
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))

simu=Vicsek(pos=pos,vel=vel,
            v=1.,
            sigma=1.,nu=3.,
            interaction_radius=R,
            box_size=L)

simu.__next__() #GPU warmup... 

#########################################################
# Without block-sparse reduction, let us compute the simulation time of 100 iterations.

simu_copy = copy.deepcopy(simu) # Make a new deepcopy
s = time.time()
for k in range(100):
    simu_copy.__next__()
e = time.time()

simulation_time = e-s

print("Average simulation time without block-sparse reduction: " + str(simulation_time) + " seconds.")

##########################################################
# Then with block-sparse reduction... First, turn on the attribute :attr:`blocksparse <sispyphe.particles.Particles.blocksparse>`. 
      
simu.blocksparse = True
      
#########################################################
# Then, we need to define the maximum number of cells. This can be set by the keyword argument ``number_of_cells`` when an instance of the class :class:`sisyphe.particles.Particles` is created. The number of cells has a strong influence on the efficiency of the method and should be chosen wisely.  When the optimal value is not known a priori, it is recommanded to use the  method :meth:`best_blocksparse_parameters() <sisyphe.particles.Particles.best_blocksparse_parameters>` which will time 100 iterations of the simulation for various numbers of cells and automatically choose the best one. Below, we test all the numbers of cells which are powers of the dimension (here :math:`d=2`) between :math:`10^2` and :math:`70^2`. 
#
 
ncell_min = 10
ncell_max = 70
fastest, nb_cells, average_simu_time, simulation_time = simu.best_blocksparse_parameters(ncell_min, ncell_max, step=1, nb_calls=100)


##################################################
# We plot the average simulation time as a function of the square root of the number of cells and print the best. 

plt.plot(nb_cells,average_simu_time)      
plt.xlabel("Square root of the number of cells") 
plt.ylabel("Simulation time") 

print("Average simulation time with block-sparse reduction: " + str(average_simu_time.min()) + " seconds.")

#################################################
# Same experiment with one million particles. 

N = 1000000
L = 100.  
R = 1.

pos = L*torch.rand((N,2)).type(dtype)
vel = torch.randn(N,2).type(dtype)
vel = vel/torch.norm(vel,dim=1).reshape((N,1))


simu=Vicsek(pos=pos,vel=vel,
            v=1.,
            sigma=1.,nu=3.,
            interaction_radius=R,
            box_size=L,
            block_sparse_reduction=False)

simu_copy = copy.deepcopy(simu) # Make a new deepcopy
s = time.time()
for k in range(100):
    simu_copy.__next__()
e = time.time()

simulation_time = e-s

print("Average simulation time without block-sparse reduction: " + str(simulation_time) + " seconds.")

##################################################
# With block-sparse reduction...

simu.blocksparse = True

fastest, nb_cells, average_simu_time, simulation_time = simu.best_blocksparse_parameters(30, 100, nb_calls=100)


##################################################
# We plot the average simulation time as a function of the square root of the number of cells and print the best. 

plt.plot(nb_cells,average_simu_time)      
plt.xlabel("Square root of the number of cells") 
plt.ylabel("Simulation time") 

print("Average simulation time with block-sparse reduction: " + str(average_simu_time.min()) + " seconds.")

###########################
# .. note::
#         The optimal parameters chosen initially may not stay optimal in the course of the simulation. This may be the case in particular if there is a strong concentration of particles.
        
##################################################
# How does it work 
# ---------------------------------
#
# Cell size and number of cells
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The cells have a rectangular shape. The length of the cells along each dimension cannot be smaller than the interaction radius :math:`R`. The maximum number of cells is thus equal to: 
# 
# .. math::
#         
#         n_\mathrm{max} = \prod_{k=1}^d \left\lfloor \frac{L_k}{R} \right\rfloor,
#
# where :math:`L_k` is the length of the (rectangular) domain along dimension :math:`k`. This corresponds to rectangular cells with a length along dimension :math:`k` equal to: 
#
# .. math:: 
#         
#         \varepsilon_k = \frac{L_k}{\left\lfloor \frac{L_k}{R} \right\rfloor}.
#       
# If the number of cells demanded :math:`n_0` exceeds :math:`n_\mathrm{max}`, this will be the chosen value. Otherwise, we first compute the typical length: 
# 
# .. math:: 
#         
#         \varepsilon_0 = \left(\frac{\prod_{k=1}^d L_k}{n_0}\right)^{1/d}
# 
# Then the length of the cells along dimension :math:`k` is set to
# 
# .. math::
#         
#         \varepsilon_k = \frac{L_k}{\left\lfloor\frac{L_k}{\varepsilon_0}\right\rfloor}.
# 
# In particular, in a square domain :math:`L_k=L` for all :math:`k` and when :math:`n_0` is a power of :math:`d`, then there are exactly :math:`n_0` square cells with length :math:`L/n_0^{1/d}`. 
# 
# 

############################################
# The block-sparse parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The initialisation or the method :meth:`best_blocksparse_parameters() <sisyphe.particles.Particles.best_blocksparse_parameters>` define three attributes which are used to speed up the computations. Given a number of cells, they are computed by the method :meth:`compute_blocksparse_parameters() <sisyphe.particles.Particles.compute_blocksparse_parameters>`.
#
# * :attr:`centroids <sisyphe.Particles.particles.centroids>` : the coordinates of the centers of the cells. 
# * :attr:`keep <sisyphe.Particles.particles.keep>` : a square BoolTensor which indicates whether two cells are contiguous. 
# * :attr:`eps <sisyphe.Particles.particles.keep>` : the length of the cells along each dimension. 
#
# The particles are clustered into the cells using the method :meth:`uniform_grid_separation() <sisyphe.toolbox.uniform_grid_separation>`. 
#
# .. note::
#     A drawback of the method is the high memory cost needed to store the boolean mask :attr:`keep <sisyphe.Particles.particles.keep>`. As a consequence, unlike the classical Verlet list method, the optimal number of cells is often **not** the maximum one. In the examples presented in this documentation, the optimal number of cells is always smaller than :math:`10^4`. 







      
