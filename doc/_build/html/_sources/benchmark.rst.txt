Benchmarks
###########

We compare 

* a `Fortran implementation <https://github.com/smotsch/Vicsek_microFlat>`_ due to Sébastien Motsch using the Verlet list method with double precision float numbers, run on the `NextGen <http://sysnews.ma.ic.ac.uk/compute-cluster/>`_ compute cluster at Imperial College London. 
* the CPU version of SiSyPHE with double precision tensors (float64) and the blocksparse-reduction method (BSR), run on an Intel MacBook Pro (2GHz Intel Core i5 processor with 8 Go of memory) ; 
* the GPU version of SiSyPHE with single precision tensors (float32) with and without the :ref:`blocksparse-reduction method <tutobsr>` (BSR), run on a `GPU cluster <http://sysnews.ma.ic.ac.uk/GPU-computing/>`_ at Imperial College London using an nVidia GTX 2080 Ti GPU ;
* the GPU version of SiSyPHE with double precision tensors (float64) with and without the :ref:`blocksparse-reduction method <tutobsr>` (BSR), run on a `GPU cluster <http://sysnews.ma.ic.ac.uk/GPU-computing/>`_ at Imperial College London using an nVidia GTX 2080 Ti GPU.

We run a :class:`Vicsek <sisyphe.models.Vicsek>` model in a square periodic box with fixed parameters :math:`L=100`, :math:`\nu=5`, :math:`\sigma=1`, :math:`c=R`, :math:`dt=0.01` and various choices of :math:`N` and :math:`R`. The simulation is run for 10 units of time (i.e. 1000 iterations). 

For each value of :math:`N`, three value of :math:`R` are tested which correspond to a dilute regime, a moderate regime and a dense mean-field regime. When the particles are uniformly scattered in the box, the average number of neighbours in the three regimes is respectively :math:`\sim3`, :math:`\sim30` and :math:`\sim300`. The regime has a strong effect on the efficiency of the Verlet list and blocksparse reduction methods. 

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

