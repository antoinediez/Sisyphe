import math
import torch
from pykeops.torch import LazyTensor


def squared_distances_tensor(x, y, L, boundary_conditions='periodic'):
    r"""Squared distances matrix as torch Tensors.

    Args:
        x ((M,d) Tensor): Rows.
        y ((N,d) Tensor): Columns.
        L ((d,) Tensor): Box size.
        boundary_conditions (list or str, optional): Boundary conditions.
            Can be one of the following:

            - list of size :math:`d` containing 0 (periodic) or 1
              (wall with reflecting boundary conditions) for each
              dimension.

            - ``"open"`` : no boundary conditions.

            - ``"periodic"`` : periodic boundary conditions.

            - ``"spherical"`` : reflecting boundary conditions on the
              sphere of radius ``L[0]/2`` and center **L**/2.

            Default is ``"periodic"``.

    Returns:
        (M,N) Tensor:

        The tensor whose :math:`(i,j)` coordinate is the squared
        distance between ``x[i,:]`` and ``y[j,:]``.

    """
    d = x.size()[1]
    x_i = x[:, None, :]  # (M,1,d)
    y_j = y[None, :, :]  # (1,N,d)
    L_k = L[None, None, :]  # (1,1,d)
    A = y_j - x_i
    if boundary_conditions == 'periodic':
        A = A + ((-L_k / 2. - A) > 0) * L_k - ((A - L_k / 2.) > 0) * L_k
    elif (type(boundary_conditions) is list
          and not (boundary_conditions == [1] * d)):
        L_k = L_k * (-torch.tensor(boundary_conditions,
                                   dtype=L.dtype,
                                   device=L.device)[None, None, :] + 1.)
        A = A + ((-L_k / 2. - A) > 0) * L_k - ((A - L_k / 2.) > 0) * L_k
    squared_dist_ij = (A ** 2).sum(-1)
    return squared_dist_ij


def lazy_xy_matrix(x, y, L, boundary_conditions='periodic'):
    r"""XY matrix as LazyTensors.

    Args:
        x ((M,d) Tensor): Rows.
        y ((N,d) Tensor: Columns.
        L ((d,) Tensor): Box size.
        boundary_conditions (list or str, optional): Boundary conditions.
            Can be one of the following:

            - list of size :math:`d` containing 0 (periodic) or 1
              (wall with reflecting boundary conditions) for each
              dimension.

            - ``"open"`` : no boundary conditions.

            - ``"periodic"`` : periodic boundary conditions.

            - ``"spherical"`` : reflecting boundary conditions on the
              sphere of radius ``L[0]/2`` and center **L**/2.

            Default is ``"periodic"``.

    Returns:
        (M,N,d) LazyTensor:

        The LazyTensor ``A`` such that for each coordinate :math:`(i,j)`,
        ``A[i,j,:]`` is the LazyTensor ``LazyTensor(y[j,:]-x[i,:])``.

    """
    d = x.size()[1]
    x_i = LazyTensor(x[:, None, :])  # (M,1,d)
    y_j = LazyTensor(y[None, :, :])  # (1,N,d)
    L_k = LazyTensor(L[None, None, :])  # (1,1,d)
    A = y_j - x_i
    if boundary_conditions == 'periodic':
        A = A + (-L_k / 2. - A).step() * L_k - (A - L_k / 2.).step() * L_k
    elif (type(boundary_conditions) is list
          and not (boundary_conditions == [1] * d)):
        L_k = L_k * LazyTensor(-torch.tensor(boundary_conditions,
                                             dtype=L.dtype,
                                             device=L.device)[None, None, :]
                               + 1.)
        A = A + (-L_k / 2. - A).step() * L_k - (A - L_k / 2.).step() * L_k
    return A


def squared_distances(x, y, L, boundary_conditions='periodic'):
    r"""Squared distances LazyTensor.

    Args:
        x ((M,d) Tensor): Rows.
        y ((N,d) Tensor: Columns.
        L ((d,) Tensor): Box size.
        boundary_conditions (list or str, optional): Boundary conditions.
            Can be one of the following:

            - list of size :math:`d` containing 0 (periodic) or 1
              (wall with reflecting boundary conditions) for each
              dimension.

            - ``"open"`` : no boundary conditions.

            - ``"periodic"`` : periodic boundary conditions.

            - ``"spherical"`` : reflecting boundary conditions on the
              sphere of radius ``L[0]/2`` and center **L**/2.

            Default is ``"periodic"``.

    Returns:
        (M,N) LazyTensor:

        The LazyTensor whose :math:`(i,j)` coordinate is the squared
        distance between ``x[i,:]`` and ``y[j,:]``.

    """
    A = lazy_xy_matrix(x, y, L, boundary_conditions)
    squared_dist_ij = (A ** 2).sum(-1)
    return squared_dist_ij


def sqdist_angles(x, y, x_orientation, L, boundary_conditions='periodic'):
    r"""Squared distances and angles LazyTensors.

    Args:
        x ((M,d) Tensor): Rows.
        y ((N,d) Tensor: Columns.
        x_orientation ((M,d) Tensor): Orientation of x.
        L ((d,) Tensor): Box size.
        boundary_conditions (list or str, optional): Boundary conditions.
            Can be one of the following:

            - list of size :math:`d` containing 0 (periodic) or 1
              (wall with reflecting boundary conditions) for each
              dimension.

            - ``"open"`` : no boundary conditions.

            - ``"periodic"`` : periodic boundary conditions.

            - ``"spherical"`` : reflecting boundary conditions on the
              sphere of radius ``L[0]/2`` and center **L**/2.

            Default is ``"periodic"``.

    Returns:
        (M,N) LazyTensor, (M,N) LazyTensor:

        The LazyTensor whose :math:`(i,j)` coordinate is the squared
        distance between ``x[i,:]`` and ``y[j,:]`` and the LazyTensor
        whose :math:`(i,j)` coordinate is the cosine of the angle
        between ``y[j,:]-x[i,:]`` and ``x_orientation[i,:]``.

    """
    A = lazy_xy_matrix(x, y, L, boundary_conditions)
    squared_dist_ij = (A ** 2).sum(-1)
    cos_ij = (A * x_orientation[:, None, :]).sum(-1) / squared_dist_ij.sqrt()
    return squared_dist_ij, cos_ij


def lazy_interaction_kernel(x, y, Rx, L, boundary_conditions,
                            vision_angle=2 * math.pi, axis=None, **kwargs):
    r"""Interaction kernel as LazyTensor.

    Args:
        x ((M,d) Tensor): Row.
        y ((N,d) Tensor): Columns.
        Rx ((N,) Tensor or float): Interaction radius of x.
        L ((d,) Tensor): Box size.
        boundary_conditions (list or str, optional): Boundary conditions.
            Can be one of the following:

            - list of size :math:`d` containing 0 (periodic) or 1
              (wall with reflecting boundary conditions) for each
              dimension.

            - ``"open"`` : no boundary conditions.

            - ``"periodic"`` : periodic boundary conditions.

            - ``"spherical"`` : reflecting boundary conditions on the
              sphere of radius ``L[0]/2`` and center **L**/2.

            Default is ``"periodic"``.
        vision_angle (float, optional): Angle of vision of x. Default is
            :math:`2\pi`.
        axis ((M,d) Tensor, optional): Orientation of **x**. Default is
            ``None``. Must be specified if **vision_angle** is not
            :math:`2\pi`.
        **kwargs: Arbitrary keywords arguments.

    Returns:
        (M,N) LazyTensor:

        The LazyTensor whose :math:`(i,j)` coordinate is 1 if ``y[j,:]``
        is in the cone of vision of ``x[i,:]`` and 0 otherwise.

    """
    if type(Rx) == type(x):
        radius = LazyTensor(Rx.reshape((x.shape[0], 1))[:, None])
    else:
        radius = Rx
    if vision_angle == 2 * math.pi:
        lazy_squared_distances = squared_distances(x, y, L,
                                                   boundary_conditions)
        K_ij = (radius ** 2 - lazy_squared_distances).step()
    else:
        sqdist_ij, cos_ij = sqdist_angles(x, y, axis, L, boundary_conditions)
        K_ij = (radius ** 2 - sqdist_ij).step() \
               * (-float(math.cos(vision_angle / 2)) + cos_ij).step()
    return K_ij


def lazy_overlapping_kernel(x, y, Rx, Ry, L, boundary_conditions, **kwargs):
    r"""Overlapping kernel as LazyTensor.

    Args:
        x ((M,d) Tensor): Rows.
        y ((N,d) Tensor): Columns.
        Rx ((M,) Tensor or float): Radius of x.
        Ry ((N,) Tenosr or float): Radius of y.
        L ((d,) Tensor): Box size.
        boundary_conditions (list or str, optional): Boundary conditions.
            Can be one of the following:

            - list of size :math:`d` containing 0 (periodic) or 1
              (wall with reflecting boundary conditions) for each
              dimension.

            - ``"open"`` : no boundary conditions.

            - ``"periodic"`` : periodic boundary conditions.

            - ``"spherical"`` : reflecting boundary conditions on the
              sphere of radius ``L[0]/2`` and center **L**/2.

            Default is ``"periodic"``.
        **kwargs: Arbitrary keywords arguments.

    Returns:
        (M,N) LazyTensor:

        The LazyTensor whose :math:`(i,j)` coordinate is 1 if ``x[i,:]``
        and ``y[j,:]`` are at a distance smaller than ``Rx[i]+Ry[j]``
        (or smaller than ``Rx+Ry`` if all the **x** and **y** particles
        have the same radius).

    Raises:
        :exc:`NotImplementerError`: if **Rx** and **Ry** do not have the
             same type, or when **Rx** and **Ry** are float but **Rx**
             is not equal to **Ry**.

    """
    if not type(Rx) == type(Ry):
        raise NotImplementedError()
    if isinstance(Rx, float) and not (Rx == Ry):
        raise NotImplementedError()
    if type(Rx) == type(x):
        R_i = LazyTensor(Rx.reshape((x.shape[0], 1))[:, None])
        R_j = LazyTensor(Ry.reshape((y.shape[0], 1))[None, :])
        R_ij = R_i + R_j
        lazy_squared_distances = squared_distances(x, y, L,
                                                   boundary_conditions)
        K_ij = (R_ij ** 2 - lazy_squared_distances).step()
        return K_ij
    else:
        return lazy_interaction_kernel(x, y, 2 * Rx, L, boundary_conditions)


def lazy_morse(x, y, Ca, la, Cr, lr, p=2, mx=1., my=1., **kwargs):
    r"""LazyTensor of the forces derived from the Morse potential.

    The Morse potential is defined by

    .. math::
        U(z) = -C_a \exp(-|z|^p / \ell_a^p) + C_r \exp(-|z|^p / \ell_r^p)

    The force exerted by a particle located in :math:`y_j` on a particle
    located on :math:`x_i` is

    .. math::
        F = - m_x m_y  \nabla U(x_i - y_j)

    where :math:`m_x` and :math:`m_y` are the masses of the particles in
    :math:`x_i` and :math:`y_j` respectively.

    Args:
        x ((M,d) Tensor): Rows.
        y ((N,d) Tensor): Columns.
        Ca (float): Attraction coefficient.
        la (float): Attraction length.
        Cr (float): Repulsion coefficient.
        lr (float): Repulsion length.
        p (int, optional): Exponent. Default is 2.
        mx ((M,) Tensor or float, optional): Mass of **x**. Default is 1.
        my ((N,) Tensor or float, optional): Mass of **y**. Default is 1.
        **kwargs: Arbitrary keywords arguments.

    Returns:
        (M,N,d) LazyTensor:

        The LazyTensor whose :math:`(i,j,:)` coordinate is the force
        exerted by ``y[j,:]`` on ``x[i,:]`` derived from the Morse
        potential.

    """
    x_i = LazyTensor(x[:, None, :])  # (M,1,D)
    y_j = LazyTensor(y[None, :, :])  # (1,N,D)
    r = x_i - y_j
    sq_norm = (r ** 2).sum(-1)  # (M,N)
    if p == 2:
        attraction = Ca / (la ** p) * (-sq_norm / (la ** 2)).exp()
        repulsion = Cr / (lr ** p) * (-sq_norm / (lr ** 2)).exp()
        nablaU = 4 * (attraction - repulsion) * r
    else:
        sq_norm = sq_norm + (-sq_norm.sign() + 1.)  # case norm=0
        attraction = Ca / (la ** p) \
                     * (-(sq_norm ** (p / 2.)) / (la ** p)).exp()
        repulsion = Cr / (lr ** p) \
                    * (-(sq_norm ** (p / 2.)) / (lr ** p)).exp()
        nablaU = (p ** 2) * (attraction - repulsion) \
                 * (sq_norm ** ((p - 2.) / 2.)) * r
    if type(mx) == type(x):
        mx_i = LazyTensor(mx.reshape((x.shape[0], 1))[:, None])
        my_j = LazyTensor(my.reshape((y.shape[0], 1))[None, :])
        return - mx_i * my_j * nablaU
    else:
        return - mx * my * nablaU


def lazy_quadratic(x, y, R, L, boundary_conditions='periodic', **kwargs):
    r"""LazyTensor of the forces derived from of a quadratic potential.

    The quadratic potential is defined by:

    .. math::
        U(z) =  \frac{1}{2R} |z|^2 - |z|

    where :math:`R` is a fixed parameter. The force exerted on a
    particle located on :math:`y_j` on a particle located on :math:`x_i`
    is

    .. math::
        F = \left(\frac{1}{R}-\frac{1}{|y_j-x_i|}\right)(y_j-x_i).

    Args:
        x ((M,d) Tensor): Rows.
        y ((N,d) Tensor): Columns.
        R ((M,) Tensor or float): Radius of the **x** particles.
        L ((d,) Tensor): Box size.
        boundary_conditions (list or str, optional): Boundary conditions.
            Can be one of the following:

            - list of size :math:`d` containing 0 (periodic) or 1
              (wall with reflecting boundary conditions) for each
              dimension.

            - ``"open"`` : no boundary conditions.

            - ``"periodic"`` : periodic boundary conditions.

            - ``"spherical"`` : reflecting boundary conditions on the
              sphere of radius ``L[0]/2`` and center **L**/2.

            Default is ``"periodic"``.
        **kwargs: Arbitrary keywords arguments.

    Returns:
        (M,N,d) LazyTensor:

        The LazyTensor whose :math:`(i,j,:)` coordinate is the force
        exerted by ``y[j,:]`` on ``x[i,:]`` derived from the quadratic
        potential.

    """
    r = lazy_xy_matrix(x, y, L, boundary_conditions)
    sq_dist = (r ** 2).sum(-1)
    sq_dist = sq_dist + (-sq_dist.sign() + 1.)  # case norm=0
    dist = sq_dist.sqrt()
    if type(x) == type(R):
        Rx = LazyTensor(R.reshape((x.shape[0], 1))[:, None])
    else:
        Rx = R
    return (1. / Rx - 1. / dist) * r


