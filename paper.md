---
title: '`SiSyPHE`: A Python package for the Simulation of Systems of interacting mean-field Particles with High Efficiency'
tags:
  - Python
  - GPU
  - particles
  - mean-field
  - self-organization
  - swarming
authors:
  - name: Antoine Diez
    affiliation: 1
affiliations:
 - name: Department of Mathematics, Imperial College London, South Kensington Campus, London, SW7 2AZ, UK
   index: 1
date: 27 June 2021
bibliography: doc/biblio_sisyphe.bib

---

# Summary

Over the past decades, the study of systems of particles has become an important 
part of many research areas, from theoretical physics to applied biology and 
computational mathematics. One of the main motivations in mathematical biology is the modelling of large
animal societies and the emergence of complex patterns from simple behavioral rules, e.g., flocks of birds, fish schools, ant colonies, etc. In the microscopic
world, particle systems are used to model a wide range of phenomena, 
from the collective motion of spermatozoa to the anarchical development of 
cancer cells. Within this perspective, there are at least three important reasons to conduct large scale computer simulations of particle systems. First, numerical experiments are essential to 
calibrate the models and test the influence of each parameter in a controlled 
environment. For instance, the renowned Vicsek model [@vicsek_novel_1995]
is a minimal model of *flocking*, which exhibits a complex behavior, studied numerically in particular in [@chate_collective_2008]. Secondly, particle simulations are used to check the validity of *macroscopic* models that describe the statistical behavior of particle systems.
These models are usually based on partial differential equations (PDE) derived 
using phenomenological considerations that are often difficult to justify 
mathematically [@degond_continuum_2008; @dimarco_self-alignment_2016; @degond_bulk_2021]. Finally, inspired by models in biology, there is an ever growing
literature on the design of algorithms based on the simulation of *artificial*
particle systems to solve tough optimization problems [@kennedy_particle_1995; @pinnau_consensus-based_2017; @totzeck_trends_2021; @grassi_particle_2020] and to construct new more efficient Markov Chain Monte Carlo methods [@del_moral_measure-valued_1998; @del_moral_mean_2013; @doucet_sequential_2001; @cappe_population_2004; @clarte_collective_2021]. The simulation of systems of particles is also at the core of molecular dynamics [@leihmkuhler], although the present library is not specifically written for this purpose. The `SiSyPHE` library builds on recent advances in hardware and software 
for the efficient simulation of large scale interacting *mean-field* particle systems, 
both on the GPU and on the CPU. The versatile object-oriented Python interface of the library is designed for the simulation and comparison of new and classical many-particle models of collective dynamics in mathematics and active matter physics, enabling ambitious numerical experiments and leading to novel conjectures and results.

# Statement of need

A major difficulty in the simulation of systems of particles is the high computational cost, typically quadratic in the number of particles, which prevents large scale experiments. The implementation of `SiSyPHE` is based on recent libraries 
originally developed for machine learning purposes to significantly accelerate 
tensor (array) computations, namely the `PyTorch` package [@paszke_pytorch_2019] and the `KeOps` library [@charlier_kernel_2021]. On a GPU, the `SiSyPHE` library speeds up both traditional Python and low-level implementations by one to three orders 
of magnitude for systems with up to several millions of particles. 

In addition, to the best of our knowledge, only model-specific packages such as @motsch_vicsek_microflat_2016 are available. The `SiSyPHE` library includes, within a common framework, the implementation of many classical models and their variants as well as recent models for which no implementation was previously available. All the models detailed in the Example gallery of the documentation are directly taken from the literature on collective dynamics in mathematics and active matter physics. Moreover, the `SiSyPHE` library is designed in such a way that new custom models can easily be added in order to facilite the study and comparison of models from a research perspective. 

The development of the `SiSyPHE` library was initially motivated by the study of *body-oriented particles* [@giacomin_alignment_2019]. 
The (formal) derivation of a macroscopic PDE model from the particle system has lead to a novel conjecture 
which postulates the existence of a class of so-called *bulk topological states* in [@degond_bulk_2021]. The quantitative comparison
between this theoretical prediction and the numerical simulation of the particle system in a suitable regime (with more than
$10^6$ particles) has confirmed the existence of these new states of matter. The study of their physical properties
which are observed in the numerical experiments but not readily explained by the PDE model is an ongoing work.

# A typical example

A typical model that is implemented in the `SiSyPHE` library is the variant of the Vicsek model
introduced by @degond_continuum_2008 and defined by the system of $2N$ Stratonovich Stochastic Differential Equations
\begin{equation}\label{eq:sde}
\mathrm{d}X^i_t = c_0 V^i_t \mathrm{d}t, \quad \mathrm{d}V^i_t = \sigma \mathsf{P}(V^i_t)\circ(J^i_t\mathrm{d}t+\mathrm{d}B^i_t),
\end{equation}
where the position at time $t$ of a particle indexed by $i\in\{1,\ldots,N\}$ is a vector $X^i_t\in\mathbb{R}^d$ and its orientation (or velocity) is a unit vector $V^i_t\in\mathbb{R}^d$ with $|V^i_t|=1$. The coefficient $c_0>0$ is the speed
of the particles (assumed to be constant), the matrix $\mathsf{P}(V^i_t)= I_d - V^i_t\otimes V^i_t$ is the orthogonal projection matrix on the plane orthgonal to $V^i_t$, $(B^i_t)^{}_t$ is an independent Brownian motion, and $\sigma>0$ is a diffusion coefficient which models the level of noise. 
The quantity $J^i_t\in\mathbb{R}^d$ is called a *target*; it is the orientation that particle $i$ is trying to adopt. 
In the Vicsek model introduced by @degond_continuum_2008, 
\begin{equation}\label{eq:target}
J^i_t = \frac{\sum_{j=1}^N K(|X^j_t-X^i_t|)V^j_t}{\big|\sum_{j=1}^N K(|X^j_t-X^i_t|)V^j_t\big|}, 
\end{equation}
where the *kernel* $K:[0,+\infty)\to[0,+\infty)$ is a smooth nonnegative
function vanishing at infinity which models the visual perception of the particles; 
in the Vicsek model, the vision of the particles depends on the distance between them. 
With the target given by \autoref{eq:target}, each particle tries to adopt the average orientation of its neighbors, which is a typical *flocking* behavior. 

On a computer, the time-continuous system given by \autoref{eq:sde} needs to be discretized first. For the Vicsek model, a natural discretization method is the (geometric) Euler-Maruyama scheme [@kloeden; @piggott]. In general, the discretization method depends on the model considered as illustrated in the Example gallery. Then, at each time step, 
the most expensive operation is the computation of the target given by \autoref{eq:target}, which requires $\mathcal{O}(N)$
operations for each of the $N$ particles. The total simulation cost is thus $\mathcal{O}(N^2T)$ where $T$ is the
total number of iterations. Within the framework of the `KeOps` library on which `SiSyPHE` is based, 
the computation of the target \autoref{eq:target} is called a *kernel operation*, which is efficiently carried out
using a *symbolic* definition of the $N\times N$ interaction matrix whose $(i,j)$-entry is $K(|X^j_t-X^i_t|)$. The computation of the target is then understood as a symbolic matrix-vector product between the interaction matrix and the vector of orientations.  


# Acknowledgements

The development of this library would not have been possible without the help of Jean Feydy, 
his constant support and precious advice. This project was initiated by Pierre Degond and 
has grown out of many discussions with him.

# References
