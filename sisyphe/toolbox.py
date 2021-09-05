import numpy as np
import torch
from pykeops.torch.cluster import sort_clusters
from pykeops.torch import LazyTensor


def volume_ball(d):
    r"""Volume of the ball of radius 1 in dimension d"""
    if np.remainder(d, 2) == 0:
        V = (np.pi ** (d / 2)) / np.math.factorial(int(d / 2))
    else:
        k = (d - 1) / 2
        V = 2 * np.math.factorial(k) * (4 * np.pi) ** k / np.math.factorial(d)
    return V


def uniform_grid_separation(x, r, L):
    r"""Return the label of the cell to which each row of **x** belongs.

    The box :math:`[0,L[0]]\times ...\times [0,L[d-1]]` is divided in
    cells of size r. The cells are numbered as follows in dimension 2:

    +--------------------+--------------------+-----+--------------------+
    |``(K[1]-1)*K[0]``   |``(K[1]-1)*K[0]+1`` | ... |``K[0]*K[1]-1``     |
    +--------------------+--------------------+-----+--------------------+
    |...                 |...                 |...  |...                 |
    +--------------------+--------------------+-----+--------------------+
    |``K[0]``            |``K[0]+1``          | ... |``2*K[0]-1``        |
    +--------------------+--------------------+-----+--------------------+
    |``0``               |``1``               | ... |``K[0]-1``          |
    +--------------------+--------------------+-----+--------------------+

    where ``K[i]`` is the number of cells along the dimension ``i``.

    Args:
        x ((N,d) Tensor): Positions.
        r ((d,) Tensor): Size of the cells at each dimension.
        L ((d,) Tensor): Box size.

    Returns:
        (N,) IntTensor:

        The Tensor of labels.

    Raises:
        :exc:`NotImplementedError`: if the dimension is :math:`d=1` or
            :math:`d>3`.

    """
    K = (L / r).floor().int()
    N, d = list(x.size())
    if d == 1 or d > 3:
        raise NotImplementedError()
    index_d = (x / r).floor().int()
    for i in range(d):
        index_d[:, i] = torch.min(index_d[:, i], K[i] - 1)
    if d == 2:
        labels = index_d[:, 0] + K[0] * index_d[:, 1]
    else:
        labels = index_d[:, 0] \
                 + K[0] * index_d[:, 1] \
                 + (K[0] * K[1]) * index_d[:, 2]
    return labels


def uniform_grid_centroids(r, L, d):
    r"""Return the coordinates of the centroids.

    The box :math:`[0,L[0]]\times ...\times [0,L[d-1]]` is divided in
    cells of size r. The cells are numbered as follows in dimension 2:

    +--------------------+--------------------+-----+--------------------+
    |``(K[1]-1)*K[0]``   |``(K[1]-1)*K[0]+1`` | ... |``K[0]*K[1]-1``     |
    +--------------------+--------------------+-----+--------------------+
    |...                 |...                 |...  |...                 |
    +--------------------+--------------------+-----+--------------------+
    |``K[0]``            |``K[0]+1``          | ... |``2*K[0]-1``        |
    +--------------------+--------------------+-----+--------------------+
    |``0``               |``1``               | ... |``K[0]-1``          |
    +--------------------+--------------------+-----+--------------------+

    where ``K[i]`` is the number of cells along the dimension ``i``.

    Args:
        r ((d,) Tensor):  Size of the cells at each dimension.
        L ((d,) Tensor): Box size.
        d (int): Dimension.

    Returns:
        (N_cell,d) Tensor:

        The Tensor of centroids: the :math:`i` component is the center
        of the cell numbered ``i``. The first dimension ``N_cell`` is
        the total number of cells (in the example above,
        ``N_cell = K[0]*K[1]``).

    Raises:
        :exc:`NotImplementedError`: if :math:`d=1` or :math:`d>3`.

    """
    if d == 1 or d > 3:
        raise NotImplementedError()
    K = (L / r).floor().int().type(torch.LongTensor).to(L.device)
    vx = torch.tensor(range(K[0]), dtype=L.dtype, device=L.device)
    vy = torch.tensor(range(K[1]), dtype=L.dtype, device=L.device)
    x = vx.repeat(K[1:].prod())
    x = x.reshape(K.prod(), 1)
    y = vy.repeat_interleave(K[0])
    if d == 3:
        y = y.repeat(K[2])
    y = y.reshape(K.prod(), 1)
    if d == 2:
        centroids = r * torch.cat((x, y), dim=1) + r / 2
    else:
        vz = torch.tensor(range(K[2]), dtype=L.dtype, device=L.device)
        z = vz.repeat_interleave(K[0] * K[1])
        z = z.reshape(K.prod(), 1)
        centroids = r * torch.cat((x, y, z), dim=1) + r / 2
    return centroids


def from_matrix(ranges_i, ranges_j, keep):
    I, J = torch.meshgrid(
        [torch.arange(0, keep.shape[0], device=keep.device),
         torch.arange(0, keep.shape[1], device=keep.device)]
    )
    redranges_i = ranges_i[
        I.t()[keep.t()]
    ]  # Use PyTorch indexing to "stack" copies of ranges_i[...]
    redranges_j = ranges_j[J[keep]]
    slices_i = (
        keep.sum(1).cumsum(0).int()
    )  # slice indices in the "stacked" array redranges_j
    slices_j = (
        keep.sum(0).cumsum(0).int()
    )  # slice indices in the "stacked" array redranges_i
    return (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)


def cluster_ranges(lab, Nlab=None):
    if Nlab is None:
        Nlab = torch.bincount(lab).float()
    pivots = torch.cat((torch.tensor([0.0], device=Nlab.device),
                        Nlab.cumsum(0)))
    return torch.stack((pivots[:-1], pivots[1:]), dim=1).int()


def block_sparse_reduction_parameters(x, y, centroids, eps, L, keep,
                                      where_dummies=False):
    r"""Compute the block sparse reduction parameters.

    Classical block sparse reduction as explained in the documentation
    of the KeOps library. The main difference is that the centroids and
    labels are computed using the :func:`uniform_grid_centroids` and the
    :func:`uniform_grid_separation` respectively. With this method a cell
    may be empty. Dummy particles are thus added to **x** and **y** at
    the positions of the centroids.

    Args:
        x ((M,d) Tensor): Rows.
        y ((N,d) Tensor): Columns
        centroids ((nb_centr,d) Tensor): Centroids.
        eps ((d,) Tensor): Size of the cells.
        L ((d,) Tensor): Size of the box.
        keep ((nb_centro,nb_centro) BoolTensor): ``keep[i,j]`` is True
            when ``centroids[i,:]`` and ``centroids[j,:]`` are
            contiguous.
        where_dummies (bool, optional): Default is False.

    Returns:

        tuple:

        The tuple of block sparse parameters:

        * **x_sorted** `((M+nb_centro,d) Tensor)`: The sorted **x_plus**
          where **x_plus** is the concatenation of **x** and the
          centroids.

        * **y_sorted** `((N+nb_centro,d) Tensor)`: The sorted **y_plus**
          where **y_plus** is the concatenation of **y** and the
          centroids.

        * **nb_centro** `(int)`: Number of centroids.

        * **labels_x** `((M+nb_centro,) IntTensor)`: The labels of
          **x_plus**.

        * **labels_y** `((n+nb_centro,) IntTensor)`: The labels of
          **y_plus**.

        * **permutation_x** `((M+nb_centro,) IntTensor)`: The
          permutation of the **x_plus** such that
          ``x_plus[permutation_x,:]`` is equal to **x_sorted**.

        * **ranges_ij**: Result of :func:`from_matrix`.

        * **map_dummies** `((M+nb_centro,N+nb_centro) LazyTensor)`:
          The :math:`(i,j)` coordinate is 0 if either ``x_sorted[i,:]``
          or ``y_sorted[j,:]`` is a centroid and 1 otherwise.
          **map_dummies** is None when **where_dummies** is False.

    """
    M, D = list(x.shape)
    N, D = list(y.shape)
    # centroids = uniform_grid_centroids(eps, L, D)
    nb_centro, D = list(centroids.size())
    Mplus = M + nb_centro

    x_plus = torch.cat((x, centroids), dim=0)
    y_plus = torch.cat((y, centroids), dim=0)

    labels_x = uniform_grid_separation(x_plus, eps, L)
    ranges_x = cluster_ranges(labels_x)
    labels_y = uniform_grid_separation(y_plus, eps, L)
    ranges_y = cluster_ranges(labels_y)
    x_sorted, labels_x_sorted = sort_clusters(x_plus, labels_x)
    u = torch.arange(0, Mplus, device=x.device).reshape(Mplus, 1)
    permutation_x, label_px = sort_clusters(u, labels_x)
    permutation_x = permutation_x[:, 0]
    y_sorted, labels_y_sorted = sort_clusters(y_plus, labels_y)
    ranges_ij = from_matrix(ranges_x, ranges_y, keep)

    if where_dummies:
        x_dummies = torch.cat(
            (torch.ones(M, dtype=x.dtype, device=x.device),
             torch.zeros(nb_centro, dtype=x.dtype, device=x.device)))
        y_dummies = torch.cat(
            (torch.ones(N, dtype=y.dtype, device=y.device),
             torch.zeros(nb_centro, dtype=y.dtype, device=y.device)))
        x_dummies_sorted, lxd = sort_clusters(x_dummies, labels_x)
        y_dummies_sorted, lyd = sort_clusters(y_dummies, labels_y)
        LTx_dummies = LazyTensor(x_dummies_sorted[:, None, None])
        LTy_dummies = LazyTensor(y_dummies_sorted[None, :, None])
        map_dummies = (LTx_dummies * LTy_dummies).sum(-1)
    else:
        map_dummies = None

    return x_sorted, y_sorted, nb_centro, labels_x, labels_y, \
           permutation_x, ranges_ij, map_dummies


def maximal_eigenvector_dim2(J):
    r"""Eigenvectors of maximum eigenvalue for symmetric matrices in
    dimension 2.

    Args:
        J ((N,2,2) Tensor): Batch of N 2D symmetric matrices.

    Returns:
        (N,2) Tensor:

        Unit eigenvector associated to the maximal eigenvalue of each
        matrix.

    """
    N = J.shape[0]
    a = J[:, 0, 0]
    b = J[:, 0, 1]
    d = J[:, 1, 1]
    sq = (b ** 2) + ((a - d) / 2.) ** 2
    ev = (a + d) / 2. + sq.sqrt()
    u = torch.cat((-b.reshape((N, 1)),
                   (a - ev).reshape((N, 1))
                   ), 1)
    normu = torch.norm(u, dim=1)
    notcool = (normu == 0.)
    nb_notcool = notcool.sum()
    if nb_notcool>0:
        u[notcool, :] = torch.randn((nb_notcool, 2),
                                    dtype=J.dtype,
                                    device=J.device)
    result = u / torch.norm(u, dim=1).reshape((N, 1))
    return result


def quat_mult(q_1, q_2):
    """Multiplication in the space of quaternions."""

    a_1, b_1, c_1, d_1 = q_1[:, 0], q_1[:, 1], q_1[:, 2], q_1[:, 3]
    a_2, b_2, c_2, d_2 = q_2[:, 0], q_2[:, 1], q_2[:, 2], q_2[:, 3]

    q_1_q_2 = torch.stack((
        a_1 * a_2 - b_1 * b_2 - c_1 * c_2 - d_1 * d_2,
        a_1 * b_2 + b_1 * a_2 + c_1 * d_2 - d_1 * c_2,
        a_1 * c_2 - b_1 * d_2 + c_1 * a_2 + d_1 * b_2,
        a_1 * d_2 + b_1 * c_2 - c_1 * b_2 + d_1 * a_2
    ), dim=1)

    return q_1_q_2


def angles_directions_to_quat(angles, directions):
    r"""Represents a list of rotation angles and axes as quaternions."""
    t = angles / 2
    return torch.cat((t.cos().view(-1, 1),
                      t.sin().view(-1, 1) * directions),
                     dim=1)
