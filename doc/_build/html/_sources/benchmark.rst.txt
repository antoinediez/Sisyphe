Benchmarks
###########

We compare 

* a `Fortran implementation <https://github.com/smotsch/Vicsek_microFlat>`_ due to Sébastien Motsch using the Verlet list method with double precision float numbers, run on the `NextGen <http://sysnews.ma.ic.ac.uk/compute-cluster/>`_ compute cluster at Imperial College London. 
* the CPU version of SiSyPHE with double precision tensors (float64) and the blocksparse-reduction method (BSR), run on an Intel MacBook Pro (2GHz Intel Core i5 processor with 8 Go of memory) ; 
* the GPU version of SiSyPHE with single precision tensors (float32) with and without the :ref:`blocksparse-reduction method <tutobsr>` (BSR), run on a `GPU cluster <http://sysnews.ma.ic.ac.uk/GPU-computing/>`_ at Imperial College London using an nVidia GTX 2080 Ti GPU ;
* the GPU version of SiSyPHE with double precision tensors (float64) with and without the :ref:`blocksparse-reduction method <tutobsr>` (BSR), run on a `GPU cluster <http://sysnews.ma.ic.ac.uk/GPU-computing/>`_ at Imperial College London using an nVidia GTX 2080 Ti GPU.

.. note::

    The choice of single or double precision tensors is automatically made according to the floating point precision of the input tensors. 

We run a :class:`Vicsek <sisyphe.models.Vicsek>` model in a square periodic box with fixed parameters :math:`L=100`, :math:`\nu=5`, :math:`\sigma=1`, :math:`c=R`, :math:`dt=0.01` and various choices of :math:`N` and :math:`R`. The simulation is run for 10 units of time (i.e. 1000 iterations). 

For each value of :math:`N`, three values of :math:`R` are tested which correspond to a dilute regime, a moderate regime and a dense mean-field regime. When the particles are uniformly scattered in the box, the average number of neighbours in the three regimes is respectively :math:`\sim3`, :math:`\sim30` and :math:`\sim300`. The regime has a strong effect on the efficiency of the Verlet list and blocksparse reduction methods. 

In the table below, the total computation times are given in seconds except when they exceed 12 hours. The best times are indicated in **bold** and the worst times in *italic*. 

+----------+---------+-----------+-------------+-------------+-------------+-------------+-------------+
|          |         | | Fortran | | sisyphe64 | | sisyphe32 | | sisyphe32 | | sisyphe64 | | sisyphe64 |
|          |         | |         | | CPU BSR   | | GPU       | | GPU BSR   | | GPU       | | GPU BSR   |
+==========+=========+===========+=============+=============+=============+=============+=============+
|          | R = 1   |    19s    |   *26s*     |             |     3.3s    |             |     3.6s    |
|          +---------+-----------+-------------+             +-------------+             +-------------+
| N = 10k  | R = 3   |   *59s*   |    29s      |             |     3.4s    |             |     3.9s    |
|          +---------+-----------+-------------+  **2.1s**   +-------------+     13s     +-------------+
|          | R = 10  |  *494s*   |    69s      |             |     3.4s    |             |     7.9s    |
+----------+---------+-----------+-------------+-------------+-------------+-------------+-------------+
|          | R = 0.3 |   309s    |  *323s*     |             |   **4.3s**  |             |     9.3s    |
|          +---------+-----------+-------------+             +-------------+             +-------------+
| N = 100k | R = 1   |  *1522s*  |   384s      |     29s     |   **4.5s**  |    973s     |      11s    |
|          +---------+-----------+-------------+             +-------------+             +-------------+
|          | R = 3   |  *3286s*  |   796s      |             |   **4.9s**  |             |      28s    |
+----------+---------+-----------+-------------+-------------+-------------+-------------+-------------+
|          | R = 0.1 |  *>12h*   |   6711s     |             |    **22s**  |             |     120s    |
|          +---------+-----------+-------------+             +-------------+             +-------------+
| N = 1M   | R = 0.3 |  *>12h*   |   6992s     |    2738s    |    **23s**  |  *>12h*     |     135s    |
|          +---------+-----------+-------------+             +-------------+             +-------------+
|          | R = 1   |  *>12h*   |   9245s     |             |    **26s**  |             |     194s    |
+----------+---------+-----------+-------------+-------------+-------------+-------------+-------------+

The GPU implementation is at least 5 times faster in the dilute regime and outperform the other methods by three orders of magnitude in the mean-field regime with large values of :math:`N`. Without the block-sparse reduction method, the GPU implementation does suffer from the quadratic complexity. The block-sparse reduction method is less sensitive to the density of particles than the traditional Verlet list method. It comes at the price of a higher memory cost (see :ref:`tutobsr`) but allows to run large scale simulations even on a CPU. On a CPU, the performances are not critically affected by the precision of the floating-point format so only simulations with double precision tensors are shown.

.. note::

    The parameters of the blocksparse reduction method are chosen using the method :meth:`best_blocksparse_parameters() <sisyphe.particles.Particles.best_blocksparse_parameters>` (see :ref:`tutobsr`). As a guideline, in this example and on the GPU, the best number of cells is respectively :math:`\sim 15^2`, :math:`\sim 42^2` and :math:`\sim 70^2` for :math:`N=10^4`, :math:`N=10^5` and :math:`N=10^6` with sisyphe32 and  :math:`\sim 25^2`, :math:`\sim 60^2` and :math:`\sim 110^2` for sisyphe64. Note that the number of cells will be automatically capped at :math:`(L/R)^2`. For the last case (sisyphe64 with :math:`R=1`), the maximal number of cells depends on the memory available. 
    
As an example, the case sisyphe32 GPU BSR with :math:`N=10^5` and :math:`R=1` is obtained with the following script. 

.. code-block:: python

    import time
    import math
    import torch
    import sisyphe.models as models
    from sisyphe.display import save

    dtype = torch.cuda.FloatTensor

    # Replace by the following line for double precision tensors.
    # dtype = torch.cuda.DoubleTensor

    N = 100000
    L = 100.
    dt = .01

    nu = 5.
    sigma = 1.

    R = 1.
    c = R

    # The type of the initial condition will determine the floating point precision.

    pos = L*torch.rand((N,2)).type(dtype)
    vel = torch.randn(N,2).type(dtype)
    vel = vel/torch.norm(vel,dim=1).reshape((N,1))

    simu = models.Vicsek(
        pos = pos,
        vel = vel,
        v = R,
        sigma = sigma,
        nu = nu,
        interaction_radius = R,
        box_size = L,
        dt = dt,
        block_sparse_reduction = True,
        number_of_cells = 42**2)

    simu.__next__() # GPU warmup

    s = time.time()
    data = save(simu, [10.], [], [])
    e = time.time()

    print(e-s)



