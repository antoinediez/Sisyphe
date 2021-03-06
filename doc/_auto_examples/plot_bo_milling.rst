
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "_auto_examples/plot_bo_milling.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__auto_examples_plot_bo_milling.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr__auto_examples_plot_bo_milling.py:


Body-oriented mill
============================================

.. GENERATED FROM PYTHON SOURCE LINES 8-12

This model is introduced in 

P. Degond, A. Diez, M. Na, Bulk topological states in a new collective dynamics model,  arXiv:2101.10864, 2021 


.. GENERATED FROM PYTHON SOURCE LINES 14-15

First of all, some standard imports. 

.. GENERATED FROM PYTHON SOURCE LINES 15-30

.. code-block:: default


    import os
    import sys
    import time
    import math
    import torch
    import numpy as np 
    from matplotlib import pyplot as plt
    import sisyphe.models as models
    from sisyphe.display import save

    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor









.. GENERATED FROM PYTHON SOURCE LINES 31-41

Body-oriented particles with initial perpendicular twist
---------------------------------------------------------------

The system is composed of body-oriented particles which are initially uniformly scattered in a periodic box but their body-orientations are "twisted". The body orientation of a particle at position :math:`(x,y,z)` is initially: 

.. math:: 

    \left(\begin{array}{ccc} 1 & 0 & 0 \\ 0 & \cos(2\pi z) & -\sin(2\pi z) \\ 0 & \sin(2\pi z) & \cos(2\pi z)\end{array}\right).



.. GENERATED FROM PYTHON SOURCE LINES 41-66

.. code-block:: default



    from sisyphe.initial import cyclotron_twist_z

    N = 1500000
    L = 1
    R = .025
    nu = 40
    c = 1
    kappa = 10

    pos, bo = cyclotron_twist_z(N,L,1,kappa,dtype)

    simu = models.BOAsynchronousVicsek(pos=pos,bo=bo,
                     v=c,
                     jump_rate=nu,kappa=kappa,
                     interaction_radius=R,
                     box_size=L,
                     boundary_conditions='periodic',
                     variant = {"name" : "normalised", "parameters" : {}},
                     options = {},
                     sampling_method='vonmises',
                     block_sparse_reduction=True,
                     number_of_cells=15**3)








.. GENERATED FROM PYTHON SOURCE LINES 67-79

Run the simulation over 5 units of time and save the azimuthal angle of the mean direction of motion defined by: 

.. math::

    \varphi = \mathrm{arg}(\Omega^1+i\Omega^2) \in [0,2\pi],

where :math:`\Omega = (\Omega^1,\Omega^2,\Omega^3)` is the mean direction of motion of the particles with velocities :math:`(\Omega_i)_{1\leq i\leq N}` : 

.. math::

    \Omega := \frac{\sum_{i=1}^N \Omega_i}{|\sum_{i=1}^N \Omega_i|}


.. GENERATED FROM PYTHON SOURCE LINES 79-86

.. code-block:: default


    frames = [5.]

    s = time.time()
    data = save(simu,frames,[],["phi"],save_file=False)
    e = time.time()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Progress:0%    Progress:1%    Progress:2%    Progress:3%    Progress:4%    Progress:5%    Progress:6%    Progress:7%    Progress:8%    Progress:9%    Progress:10%    Progress:11%    Progress:12%    Progress:13%    Progress:14%    Progress:15%    Progress:16%    Progress:17%    Progress:18%    Progress:19%    Progress:20%    Progress:21%    Progress:22%    Progress:23%    Progress:24%    Progress:25%    Progress:26%    Progress:27%    Progress:28%    Progress:29%    Progress:30%    Progress:31%    Progress:32%    Progress:33%    Progress:34%    Progress:35%    Progress:36%    Progress:37%    Progress:38%    Progress:39%    Progress:40%    Progress:41%    Progress:42%    Progress:43%    Progress:44%    Progress:45%    Progress:46%    Progress:47%    Progress:48%    Progress:49%    Progress:50%    Progress:51%    Progress:52%    Progress:53%    Progress:54%    Progress:55%    Progress:56%    Progress:57%    Progress:58%    Progress:59%    Progress:60%    Progress:61%    Progress:62%    Progress:63%    Progress:64%    Progress:65%    Progress:66%    Progress:67%    Progress:68%    Progress:69%    Progress:70%    Progress:71%    Progress:72%    Progress:73%    Progress:74%    Progress:75%    Progress:76%    Progress:77%    Progress:78%    Progress:79%    Progress:80%    Progress:81%    Progress:82%    Progress:83%    Progress:84%    Progress:85%    Progress:86%    Progress:87%    Progress:88%    Progress:89%    Progress:90%    Progress:91%    Progress:92%    Progress:93%    Progress:94%    Progress:95%    Progress:96%    Progress:97%    Progress:98%    Progress:99%    Progress:100%



.. GENERATED FROM PYTHON SOURCE LINES 87-88

Print the total simulation time and the average time per iteration. 

.. GENERATED FROM PYTHON SOURCE LINES 88-94

.. code-block:: default



    print('Total time: '+str(e-s)+' seconds')
    print('Average time per iteration: '+str((e-s)/simu.iteration)+' seconds')






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total time: 1655.7933239936829 seconds
    Average time per iteration: 0.08278966619968414 seconds




.. GENERATED FROM PYTHON SOURCE LINES 95-96

Plot the azimuthal angle :math:`\varphi`. 

.. GENERATED FROM PYTHON SOURCE LINES 96-101

.. code-block:: default


    plt.plot(data["time"],data["phi"])
    plt.xlabel("time")
    plt.ylabel("Azimuthal angle")




.. image:: /_auto_examples/images/sphx_glr_plot_bo_milling_001.png
    :alt: plot bo milling
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    Text(55.847222222222214, 0.5, 'Azimuthal angle')




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 27 minutes  35.923 seconds)


.. _sphx_glr_download__auto_examples_plot_bo_milling.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_bo_milling.py <plot_bo_milling.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_bo_milling.ipynb <plot_bo_milling.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
