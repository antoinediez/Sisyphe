===============================
Background and motivation
===============================

.. image:: _static/corridor.gif
    :scale: 100% 
    :alt: Corridor
    :align: center

Scope of application
=======================


Over the past decades, the study of systems of particles has become an important part of many research areas, from theoretical physics to applied biology and computational mathematics. Some current research trends include the following. 

* Since the 80's, there is a joint effort from biologists, physicists, computer graphics scientists and more recently mathematicians to propose accurate models for large **self-organized** animal societies. While the motivations of the different communities are largely independent, a commmon goal is to be able to explain and reproduce the **emergence of complex patterns from simple interaction rules**. Typical examples of complex systems include `flocks of birds <https://en.wikipedia.org/wiki/Flock_(birds)>`_, `fish schools <https://en.wikipedia.org/wiki/Shoaling_and_schooling>`_, `herds of mammals <https://www.youtube.com/watch?v=5IFLz4CETj4>`_ or `ant colonies <https://en.wikipedia.org/wiki/Ant_colony>`_. It seems reasonable to consider that none of these animals has an accute consciousness of the whole system organization but rather follows simple behavioral patterns: stay close to the group, do not collide with another individual, copy the attitude of the close neighbours etc. These models are often called **swarming models** or **collective dynamics** models :cite:`vicsek_collective_2012, degond_mathematical_2018, albi_vehicular_2019, naldi_particle_2010`.

* The microscopic world if full of complex systems made of many simple entities. Colonies of bacteria :cite:`wensink_meso-scale_2012` are a natural example reminiscent of the macroscopic animal societies described above. But complex patterns can also emerge from **systems of non-living particles**. Our body is actually a very complex self-organized assembly of cells which dictate every aspect of our life: our brain works thanks to the communications between billions of neurons :cite:`fournier_toy_2016, andreis_mckeanvlasov_2018`, the reproduction is based on the competition between millions of spermatozoa :cite:`creppy_turbulence_2015` and sometimes, death is sadly caused by the relentless division of cancer cells. Unlike the previous models, the communication between the particles cannot be based on their visual perception, but rather on chemical agents, on physical (geometrical) constraints etc. 

* On a completely different side, there is an ever growing number of methods in computational mathematics which are based on the simulation of systems of particles. The pioneering `Particle Swarm Optimization method <https://en.wikipedia.org/wiki/Particle_swarm_optimization>`_ :cite:`kennedy_particle_1995` has shown how biologically-inspired artificial systems of particles can be used to solve tough optimization problems. Following these ideas, other **Swarm Intelligence** optimization algorithms have been proposed up to very recently :cite:`pinnau_consensus-based_2017, totzeck_numerical_2018, grassi_particle_2020, totzeck_trends_2021`. Since the beginning of the 2000's, particle systems are also at the core of `filtering <https://en.wikipedia.org/wiki/Particle_filter>`_ :cite:`crisan_survey_2002, doucet_sequential_2001, kantas_overview_2009, del_moral_feynman-kac_2004, del_moral_mean_2013` and `sampling <https://en.wikipedia.org/wiki/Monte_Carlo_method>`_ methods :cite:`cappe_population_2004, clarte_collective_2021`. Recently, a particle-based interpretation :cite:`mei_mean_2018, chizat_global_2018, rotskoff_trainability_2019, de_bortoli_quantitative_2020, sirignano_mean_2020` of the training task of neural networks has lead to new theoretical convergence results.

Why do we need to simulate particle systems?
==================================================

Beyond the self-explanatory applications for particle-based algorithms in computational mathematics, the simulation of systems of particles is also a crucial **modelling tool** in Physics and Biology. 

* On the one hand, field experiments are useful to collect data and trigger new modelling ideas. On the other hand, numerical simulations become necessary to test these ideas and to calibrate the models. **Numerical experiments** can be conducted to identify which mechanisms are able to produce a specific phenomena in a **controlled environment** :cite:`chate_collective_2008`. 

* On a more theoretical side, the direct mathematical study of particle systems can rapidly become incredibly difficult. Inspired by the `kinetic theory of gases <https://en.wikipedia.org/wiki/Kinetic_theory_of_gases>`_, many **mesoscopic** and **macroscopic** models have been proposed to model the average statistical behavior of particle systems rather than the individual motion of each particle :cite:`toner_flocks_1998, degond_continuum_2008, naldi_particle_2010`. Within this framework, a particle system is rather seen as a **fluid** and it is decribed by `Partial Differential Equations <https://en.wikipedia.org/wiki/Partial_differential_equation>`_ (PDE) reminiscent from the theory of `fluid dynamics <https://en.wikipedia.org/wiki/Fluid_dynamics>`_. PDE models are more easily theoretically and numerically tractable. However, **cheking the validity** of these models is not always easy and is sometimes only postulated based on phenomenological considerations. In order to design good models, it is often necessary to go back-and-forth between the PDE models and the numerical simulation of the underlying particle systems. 

The development of the SiSyPHE library was initially motivated by the study of :class:`body-oriented particles <sisyphe.particles.BOParticles>` :cite:`giacomin_alignment_2019`. The (formal) derivation of a macroscopic PDE model from the particle system has lead to a novel conjecture which postulates the existence of a class of so-called **bulk topological states** :cite:`degond_bulk_2021`. The quantitative comparison between this theoretical prediction and the `numerical simulation of the particle system <https://figshare.com/projects/Bulk_topological_states_in_a_new_collective_dynamics_model/96491>`_ in a suitable regime (with more than :math:`10^6` particles) has confirmed the existence of these new states of matter. The study of their physical properties which are observed in the numerical experiments but not readily explained by the PDE model is an ongoing work.


Mean-field particle systems
==============================================

Currently, the models implemented in the SiSyPHE library belong to the family of **mean-field models**. It means that the motion of each particle is influenced by the average behavior of the whole system. The `Vicsek model <https://en.wikipedia.org/wiki/Vicsek_model>`_ :cite:`vicsek_novel_1995, degond_continuum_2008` and the Cucker-Smale model :cite:`cucker_mathematics_2007, ha_emergence_2009, naldi_particle_2010` are two popular examples of mean-field models where each particle tries to move in the average direction of motion of its neighbours (it produces a so-called *flocking* behavior). 


The mathematical point of view   
--------------------------------

From a mathematical point of view, a mean-field particle system with :math:`N` particles is defined as a Markov process in :math:`(\mathbb{R}^d)^N` with a generator :math:`\mathcal{L}_N` whose action on a test function :math:`\varphi_N` is of the form:

.. math::

    \mathcal{L}_N\varphi_N(x^1,\ldots,x^N) = \sum_{i=1}^N L_{\mu_{\mathbf{x}^N}}\diamond_i \varphi_N (x^1,\ldots,x^N),
    
where :math:`\mathbf{x}^N = (x^1,\ldots,x^N)\in (\mathbb{R}^d)^N` and

.. math::

    \mu_{\mathbf{x}^N} := \frac{1}{N}\sum_{i=1}^N \delta_{x^i},

is the so-called **empirical measure**. The operator :math:`L_{\mu_{\mathbf{x}_N}}` depends on the empirical measure and acts on one-variable test functions on :math:`\mathbb{R}^d`. Given an operator :math:`L` and a test function :math:`\varphi_N`, the notation :math:`L\diamond_i \varphi_N` denotes the function 

.. math::

    L\diamond_i \varphi_N : (x^1,\ldots,x^N) \in (\mathbb{R}^d)^N \mapsto L[x\mapsto\varphi_N(x_1,\ldots,x^{i-1},x,x^{i+1},\ldots,x^N)](x^i)\in \mathbb{R}.
    
    
The dynamics of the mean-field system depends on the operator :math:`L_{\mu_{\mathbf{x}_N}}` which is typically either a `diffusion operator <https://en.wikipedia.org/wiki/Diffusion_process>`_ :cite:`degond_continuum_2008, ha_emergence_2009` or a `jump operator <https://en.wikipedia.org/wiki/Jump_process>`_ :cite:`dimarco_self-alignment_2016, andreis_mckeanvlasov_2018`. In the first case, the mean-field particle systems can be alternatively defined by a system of :math:`N` coupled `Stochastic Differential Equations <https://en.wikipedia.org/wiki/Stochastic_differential_equation>`_ of the form: 

.. math::

    \mathrm{d}X^i_t = b(X^i_t,\mu_{\mathcal{X}^N_t})\mathrm{d}t + \sigma(X^i_t,\mu_{\mathcal{X}^N_t})\mathrm{d} B^i_t,\quad i\in\{1,\ldots,N\},

.. math::

    \mu_{\mathcal{X}^N_t} := \frac{1}{N}\sum_{i=1}^N \delta_{X^i_t},
    
where :math:`(B^i_t)_t` are :math:`N` independent Brownian motions and the coefficients :math:`b` and :math:`\sigma` are called the *drift* and *diffusion* coefficients. In most cases, these coefficients are *linear* or *quasi-linear* which means that they are of the form 

.. math::

    b(X^i_t,\mu_{\mathcal{X}^N_t}) = \tilde{b}{\left(X^i_t, \frac{1}{N}\sum_{j=1}^N K(X^i_t,X^j_t)\right)}, 

for two given functions :math:`\tilde{b}:\mathbb{R}^d\times\mathbb{R}^n\to\mathbb{R}^d` and :math:`K:\mathbb{R}^d\times\mathbb{R}^d\to \mathbb{R}^n`. 

When the particles are initially statistically independent, it can be shown that when :math:`N\to+\infty`, each particle :math:`X^i_t` converges towards an independent copy of the solution of the so-called **McKean-Vlasov** diffusion process :cite:`mckean_propagation_1969, sznitman_topics_1991, meleard_asymptotic_1996` defined by the Stochastic Differential Equation 

.. math::

    \mathrm{d}\overline{X}_t = b(\overline{X}_t,f_t)\mathrm{d}t + \sigma(\overline{X}_t,f_t)\mathrm{d} B_t,
    
where :math:`(B_t)_t` is a Brownian motion and :math:`f_t` is the law of the process :math:`\overline{X}_t`. It satisfies the **Fokker-Planck** Partial Differential Equation 

.. math::

    \partial_t f_t = - \nabla\cdot(b(x,f_t)f_t) + \frac{1}{2}\sum_{i,j=1}^N \partial_{x_i}\partial_{x_j}(a_{ij}(x,f_t)f_t),
    
with :math:`x=(x_1,\ldots,x_d)\in \mathbb{R}^d` and :math:`a=\sigma\sigma^\mathrm{T}`. This phenomenon is called **propagation of chaos**, following the terminology introduced by Kac in the 50's :cite:`kac_foundations_1956, mckean_propagation_1969, sznitman_topics_1991, meleard_asymptotic_1996, hauray_kacs_2014`. 


.. note::
    
    A popular example of mean-field particle system is the (stochastic) Cucker-Smale model :cite:`cucker_mathematics_2007, ha_emergence_2009, naldi_particle_2010`. Each particle is defined by its position :math:`X^i_t \in\mathbb{R}^d` and its velocity :math:`V^i_t\in\mathbb{R}^d` which evolve according to the system of :math:`2N` Stochastic Differential Equations: 
    
    .. math::
    
        \mathrm{d}X^i_t = V^i_t\mathrm{d}t,\quad \mathrm{d}V^i_t = \frac{1}{N}\sum_{i=1}^N K(|X^j_t-X^i_t|)(V^j_t-V^i_t)\mathrm{d}t + \mathrm{d}B^i_t,
        
    where :math:`K:[0,+\infty)\to[0,+\infty)` is an **observation kernel** which models the visual perception of the particles. In this example, the function :math:`K` is a smooth function vanishing at infinity and the communication between the particles is based on the distance between them. The motion of the particles follows the Newton's laws of motion with an additional stochastic term. The term :math:`V^j_t-V^i_t` is a relaxation force (for a quadratic potential) which tends to align the velocities of particle :math:`i` and particle :math:`j`. 
        


Simulating mean-field particle systems
------------------------------------------------------

On a computer, the above examples which are time-continuous needs to be discretized, using for instance `one of the classical numerical schemes <https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method>`_ for Stochastic Differential Equations. Then the difficulty lies in the evaluation of the empirical measure **at each time-step** of the numerical scheme. 

In the example of the Cucker-Smale model, the force exerted on a particle is the sum of :math:`N` small relaxation forces of order :math:`1/N`. The total number of operations required is thus of order :math:`\mathcal{O}(N)` **for each particle**. Since there are :math:`N` particles, the total time complexity of the algorithm is thus :math:`\mathcal{O}(N^2T)` where :math:`T` is the total number of iterations of the numerical scheme. 

This **quadratic cost** is the main bottleneck in the simulation of mean-field particle systems. As explained in `the documentation of the KeOps library <https://www.kernel-operations.io/keops/introduction/why_using_keops.html>`_, the evaluation of the :math:`N` forces at time :math:`t`

.. math::
    
    F_i(t) = \frac{1}{N}\sum_{i=1}^N K(|X^j_t-X^i_t|)(V^j_t-V^i_t), \quad i\in\{1,\ldots,N\},
    
is called a **kernel operation** and can be understood as a discrete convolution operation (or matrix-vector product) between the matrix of distances :math:`(K_{ij})_{i,j\in\{1\,\ldots,N\}}` where :math:`K_{ij} = K(|X^j_t-X^i_t|)` and the vector of velocities :math:`(V^j_t-V^i_t)_{j\in\{1,\ldots, N\}}`. When :math:`N` is large (say :math:`N>10^4`), such operation is too costly even for array-based programming languages such as Matlab: the :math:`N\times N` kernel matrix :math:`(K_{ij})_{i,j}` would simply not fit into the memory. With lower level languages (Fortran, C), this operation can be implemented more efficiently with two nested loops but with a significantly higher global coding effort and with less versatility. 

Over the past decades, several workarounds have been proposed. Popular methods include 

* the `low-rank decomposition <https://en.wikipedia.org/wiki/Low-rank_matrix_approximations>`_ of the kernel matrix, 

* the fast-multipole methods :cite:`greengard_fast_1987` which to treat differently short- and long-range interactions, 

* the `Verlet list method <https://en.wikipedia.org/wiki/Verlet_list>`_ which is based on a grid decompostion of the spatial domain to reduce the problem to only short-range interactions between subsets of the particle system, 

* the Random Batch Method :cite:`jin_random_2019` which is based on a stochastic approximation where only interactions between randomly sampled subsets (*batches*) of the particle system are computed. 

All these methods require an significant amount of work, either in terms of code or to justify the approximation procedures. 


The SiSyPHE library
========================

The present implementation is based on recent libraries originally developed for machine learning purposes to significantly accelerate such tensor (array) computations, namely the `PyTorch <https://github.com/pytorch/pytorch>`_ package and the `KeOps <https://www.kernel-operations.io/keops/index.html>`_ library :cite:`charlier_kernel_2021`. Using the KeOps framework, the kernel matrix is a **symbolic matrix** defined by a mathematical formula and no approximation is required in the computation of the interactions (up to the time discretization). The SiSyPHE library speeds up both traditional Python and low-level CPU implementations by **one to three orders of magnitude** for systems with up to several millions of particles. Although the library is mainly intended to be used on a GPU, the implementation is fully functional on the CPU with a significant gain in efficiency.   

Moreover, the **versatile object-oriented Python interface** is well suited to the comparison and study of new and classical many-particle models. This aspect is fundamental in applications in order to conduct ambitious numerical experiments in a systematic framework, even for particles with a complex structure and with a significantly reduced computational cost

References
===============

.. bibliography::



