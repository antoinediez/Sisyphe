===============================
Installation
===============================


Requirements
================

- **Python 3** with packages **NumPy** and **SciPy** 
- **PyTorch** : version>= 1.5
- **PyKeops** : version>= 1.5

Installation
============= 

Using pip
------------

In a terminal, type:

.. code-block:: bash

    pip install sisyphe
    
On Google Colab
-------------------

The easiest way to get a working version of SiSyPHE is to use the free virtual machines provided by `Google Colab <https://colab.research.google.com>`_.

1. On a new Colab notebook, navigate to Edit→Notebook Settings and select GPU from the Hardware Accelerator drop-down.

2. Install PyKeops with the Colab specifications **first** by typing
    
    .. code-block:: bash

        !pip install pykeops[colab]
    
3. Install SiSyPHE by typing 
    
    .. code-block:: bash

        !pip install sisyphe
    
From source
-------------------

Alternatively, you can clone the `git repository <https://github.com/antoinediez/Sisyphe>`_ at a location of your choice. 


Testing the installation
============================

The following test function will check the configuration and run the simulation of a system of body-oriented particles (see the example gallery). This simulation uses the main features of the SiSyPHE library: a complex system, nontrivial boundary conditions, a sampling-based interaction mechanism, the blocksparse reduction method and a nontrivial initial condition. Moreover, this model is `theoretically well-understood <https://arxiv.org/abs/2101.10864>`_ which provides a theoretical baseline to check the accuracy of the output of the simulation. 

.. warning::
    This function is mainly intended to be runned on a GPU. Running this function on a CPU will take a long time! See below for a quick testing procedure. 

In a Python terminal, type 

.. code-block:: python

    import sisyphe
    sisyphe.test_sisyphe()
    
On a fresh environment, it should return

.. code-block:: console

    Welcome! This test function will create a system of body-oriented particles in a ``milling configuration'' (cf. the example gallery). The test will be considered as successful if the computed milling speed is within a 5% relative error range around the theoretical value.

     Running test, this may take a few minutes...

     Check configuration... 
    [pyKeOps] Initializing build folder for dtype=float32 and lang=torch in /root/.cache/pykeops-1.5-cpython-37 ... done.
    [pyKeOps] Compiling libKeOpstorch180bebcc11 in /root/.cache/pykeops-1.5-cpython-37:
           formula: Sum_Reduction(SqNorm2(x - y),1)
           aliases: x = Vi(0,3); y = Vj(1,3); 
           dtype  : float32
    ... 
    [pyKeOps] Compiling pybind11 template libKeOps_template_574e4b20be in /root/.cache/pykeops-1.5-cpython-37 ... done.
    Done.

    pyKeOps with torch bindings is working!

    Done.

     Sample an initial condition... 
    Done.

     Create a model... 
    Done.

     Run the simulation... 
    [pyKeOps] Compiling libKeOpstorch269aaf150e in /root/.cache/pykeops-1.5-cpython-37:
           formula: Sum_Reduction((Step((Var(5,1,2) - Sum(Square((((Var(0,3,1) - Var(1,3,0)) + (Step(((Minus(Var(2,3,2)) / Var(3,1,2)) - (Var(0,3,1) - Var(1,3,0)))) * Var(2,3,2))) - (Step(((Var(0,3,1) - Var(1,3,0)) - (Var(2,3,2) / Var(4,1,2)))) * Var(2,3,2))))))) * Var(6,16,1)),0)
           aliases: Var(0,3,1); Var(1,3,0); Var(2,3,2); Var(3,1,2); Var(4,1,2); Var(5,1,2); Var(6,16,1); 
           dtype  : float32
    ... 
    Done.
    Progress:100%Done.

     Check the result... 
    Done.

     SiSyPHE is working!    

    
The core functionalities of the library are automatically and continuously tested through a GitHub workflow based on the module :mod:`sisyphe.test.quick_test`. The testing functions include basic computations on simple examples (computation of simple local averages in various situations) and small scales simulations. Note that unlike the function :meth:`sisyphe.test_sisyphe()`, these testing functions do not check the accuracy of the output of the simulations but only check that the code runs without errors. It is possible to use the `Pytest package <https://docs.pytest.org/en/6.2.x/>`_ to run these tests manually: on a Python terminal, type

.. code-block:: python

    import pytest
    from sisyphe.test import quick_test
    retcode = pytest.main([quick_test.__file__,])


    