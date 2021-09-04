import math
import torch
import sisyphe.models as models
import sisyphe.particles as particles
from sisyphe.toolbox import block_sparse_reduction_parameters
from sisyphe.display import save

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def test_neighbours():
    """Test the computation of the number of neighbours for a simple
    configuration with various boundary conditions and vision angles.
    """
    L = 10.
    R = 1.
    N = 9
    pos = torch.tensor([[5., 5.],
                        [.1, .1],
                        [5.5, 5.],
                        [4.5, 5.],
                        [5., 4.5],
                        [5., 5.5],
                        [9.9, .1],
                        [9.9, 9.9],
                        [.1, 9.9]
                       ]).type(dtype)

    simu = particles.Particles(
         pos=pos,
         interaction_radius=R,
         box_size=L)

    n_neigh = simu.number_of_neighbours()
    assert int(n_neigh[0]) == 5
    assert int(n_neigh[1]) == 4

    simu.bc = [0,1]
    n_neigh = simu.number_of_neighbours()
    assert int(n_neigh[1]) == 2

    if use_cuda:
        axis = torch.randn(N,2).type(dtype)
        axis = axis/torch.norm(axis, dim=1).reshape((N, 1))
        axis[0,:] = torch.tensor([1., 0.]).type(dtype)
        simu.axis = axis
        simu.angle = math.pi - .01
        n_neigh = simu.number_of_neighbours()
        assert int(n_neigh[0]) == 2


def test_bsr():
    """Test the labelling of the particles for the block sparse
    reduction method."""
    L = 9.
    R = .1
    N = 9
    pos = torch.tensor([[4.6, 1.5],
                        [1.6, 1.5],
                        [7.6, 1.5],
                        [1.6, 4.5],
                        [4.6, 4.5],
                        [7.6, 4.5],
                        [1.6, 7.5],
                        [4.6, 7.5],
                        [7.6, 7.5]
                       ]).type(dtype)

    simu = particles.Particles(
         pos=pos,
         interaction_radius=R,
         box_size=L,
         block_sparse_reduction=True,
         number_of_cells=9
    )

    (x_sorted,
     y_sorted,
     nb_centro,
     labels_x,
     labels_y,
     permutation_x,
     ranges_ij,
     map_dummies) = block_sparse_reduction_parameters(
        simu.pos, simu.pos, simu.centroids, simu.eps, simu.L, simu.keep,
        where_dummies=False)

    assert int(labels_x[0]) == 1
    assert int(labels_x[1]) == 0
    assert int(labels_x[2]) == 2
    assert int(labels_x[3]) == 3
    assert int(labels_x[4]) == 4
    assert int(labels_x[5]) == 5
    assert int(labels_x[6]) == 6
    assert int(labels_x[7]) == 7
    assert int(labels_x[8]) == 8


def test_simu():
    """Run a simulation on a small scale example."""
    N = 10000
    L = 100.
    dt = .01
    nu = 5.
    sigma = 1.
    R = 3.
    c = R
    pos = L * torch.rand((N, 2)).type(dtype)
    vel = torch.randn(N, 2).type(dtype)
    vel = vel / torch.norm(vel, dim=1).reshape((N, 1))
    simu = models.Vicsek(
        pos=pos,
        vel=vel,
        v=c,
        sigma=sigma,
        nu=nu,
        interaction_radius=R,
        box_size=L,
        dt=dt,
        block_sparse_reduction=True,
        number_of_cells=42 ** 2)
    data = save(simu, [.1], [], [])
