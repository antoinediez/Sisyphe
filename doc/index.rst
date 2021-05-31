.. Sisyphe documentation master file, created by
   sphinx-quickstart on Thu Apr 29 16:21:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Simulation of Systems of Particles with High Efficiency
==============================================================================

The SiSyPHE library builds on recent advances in hardware and software for the efficient simulation of **large scale interacting particle systems**, both on the **GPU** and on the CPU. The implementation is based on recent libraries originally developed for machine learning purposes to significantly accelerate tensor (array) computations, namely the `PyTorch <https://github.com/pytorch/pytorch>`_ package and the `KeOps <https://www.kernel-operations.io/keops/index.html>`_ library. The **versatile object-oriented Python interface** is well suited to the comparison of new and classical many-particle models, enabling ambitious numerical experiments and leading to novel conjectures. The SiSyPHE library speeds up both traditional Python and low-level implementations by **one to three orders of magnitude**. 

The project is hosted on `GitHub <https://github.com/antoinediez/Sisyphe>`_, under the permissive `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_.

Author
--------

`Antoine Diez <https://antoinediez.gitlab.io/>`_, Imperial College London 

.. toctree::
   :maxdepth: 2

   benchmark.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials and examples

   _auto_tutorials/index
   _auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API

   api/API_particles
   api/API_models
   api/API_kernels
   api/API_sampling
   api/API_display
   api/API_toolbox
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
