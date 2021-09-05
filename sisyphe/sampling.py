import math
import torch
from .toolbox import angles_directions_to_quat, quat_mult


def uniform_sphere_rand(N, d, device, dtype=torch.float32):
    r"""Uniform sample on the unit sphere :math:`\mathbb{S}^{d-1}`.

    Args:
        N (int): Number of samples.
        d (int): Dimension
        device (torch.device): Device on which the samples are created
        dtype (torch.dtype, optional): Default is torch.float32

    Returns:
        (N,D) Tensor:

        Sample.

    """
    V = torch.randn(N, d, device=device, dtype=dtype)
    V = V / torch.norm(V, dim=1).reshape(N, 1)
    if (torch.isnan(V)).sum() > 0:
        return uniform_sphere_rand(N, d, device, dtype=dtype)
    else:
        return V


def uniform_angle_rand(mu, eta):
    r"""Uniform sample in an angle interval in dimension 2.

    Add a uniform angle in :math:`[-\pi\eta,\pi\eta]` to the angle of
    **mu**.

    Args:
        mu ((N,2) Tensor): Center of the distribution for each of the
            N samples.
        eta ((N,2) Tensor or float): Deviation for each of the N samples.

    Returns:
        (N,2) Tensor:

        Sample.

    """
    N, D = list(mu.size())
    angle_mu = torch.atan2(mu[:, 1], mu[:, 0])
    u = 2 * math.pi * eta * torch.rand(N,
                                       dtype=mu.dtype,
                                       device=mu.device) - math.pi * eta
    sample_angle = angle_mu + u
    result = torch.cat(
        (torch.cos(sample_angle).reshape(N, 1),
         torch.sin(sample_angle).reshape(N, 1)),
        dim=1)
    return result


""" Rejection sampler for the von Mises Fisher distribution 
on the (D-1)-sphere using the Ulrich-Wood algorithm
"""

def sample_W(kappa,N,d):
    r"""Sample the first coordinate of a von Mises distribution with
    center at :math:`(1,0,...0)^T` and concentration **kappa**.

    Args:
        kappa ((N,) Tensor): Concentration parameter.
            Also accept (1,) Tensor if all the samples have the same
            concentration parameter.
        N (int): Number of samples. Must be equal to the size of kappa
            when the size of kappa is >1.
        d: Dimension.

    Returns:
        (N,) Tensor:

        Random sample in :math:`\mathbb{R}`.

    """
    if N == 0:
        return torch.tensor([],device=kappa.device)
    b = (-2 * kappa + torch.sqrt(4 * (kappa ** 2) + (d - 1) ** 2)) / (d - 1)
    x0 = (1-b) / (1+b)
    c = kappa * x0 + (d - 1) * torch.log(1 - x0 ** 2)
    X = kappa.new(2 * N, d - 1).normal_()
    X = (X**2).sum(1)
    X0 = X[:N]
    X1 = X[N:]
    beta = X0 / (X0 + X1)   #beta is a Beta((D-1)/2,(D-1)/2)
    w = (1 - (1+b) * beta) / (1 - (1-b) * beta)
    W = w
    U = kappa.new(int(N)).uniform_()
    reject = kappa * w + (d - 1) * torch.log(1 - x0 * w) - c < torch.log(U)
    Nreject = reject.sum().item()
    if kappa.shape[0] == 1:
        W[reject] = sample_W(kappa, Nreject, d)
    else:
        W[reject] = sample_W(kappa[reject], Nreject, d)
    return W


def vonmises_rand(mu,kappa):
    r"""Von Mises sample on the sphere :math:`\mathbb{S}^{d-1}` using
    the Ulrich-Wood algorithm [W1994]_.

    .. [W1994] A. Wood, Simulation of the von Mises Fisher distribution,
       *Comm. Statist. Simulation Comput.*, Vol. 23, No. 1 (1994).

    Args:
        mu ((N,d) Tensor): Centers of the N samples.
        kappa: ((N,) Tensor or (1,) Tensor): Concentration parameter.

    Returns:
        (N,d) Tensor:

        Sample with center **mu** and concentration parameter **kappa**.

    """
    N, D = list(mu.size())
    W = sample_W(kappa, N, D)
    # Uniformly sampled unit random vector V in the (D-2)-sphere
    if D == 2:
        u = torch.rand((N, 1), dtype=mu.dtype, device=mu.device) < .5
        V = 2. * u - 1.
    else:
        V = torch.randn((N, D - 1), dtype=mu.dtype, device=mu.device)
        V = V / torch.norm(V, dim=1).reshape((N, 1))
    # Construct a von Mises random variable X with center (1,0,0...)^T
    X = torch.cat((W[:, None], torch.sqrt((1 - W[:, None] ** 2)) * V), 1)
    if D == 2:
        angle_mu = torch.atan2(mu[:, 1], mu[:, 0])
        X_angle = torch.atan2(X[:, 1], X[:, 0])
        X_angle = X_angle + angle_mu
        result = torch.cat(
            (torch.cos(X_angle).reshape(N, 1),
             torch.sin(X_angle).reshape(N, 1)),
            dim=1)
    else:
        # Construct the Householder matrix Q associated to mu
        e1 = torch.zeros(D, dtype=mu.dtype, device=mu.device)
        e1[0] = 1
        u = mu - e1[None, :]
        v = u / torch.norm(u, dim=1).reshape((N, 1))
        v[torch.isnan(v)] = 0
        Q = torch.eye(D, dtype=mu.dtype, device=mu.device) - \
            2 * torch.einsum('ijl,ilk->ijk', v[:, :, None], v[:, None, :])
        # Multiply the Householder matrix with X to get the correct result
        result = torch.einsum('ijl,ilk->ijk', Q, X[:, :, None])
        result = result.reshape(N, D)
    return result

#### Rejection sampler: uniform quaternion in a ball around Id ####

def sample_angles(N, scales):
    r"""Sample angles by rejection sampling."""
    # if scales.size()==torch.Size([1]):
    #     s = scales
    # else:
    #     s = scales.reshape(N,1)
    angles = scales * torch.rand(N, dtype=scales.dtype, device=scales.device)
    uniform = torch.rand(N, dtype=scales.dtype, device=scales.device)

    threshold = (1 - angles.cos()) / (1 - scales.cos())
    reject = (uniform > threshold).view(-1)

    M = int(reject.sum())

    if M == 0:
        return angles
    else:
        if scales.size() == torch.Size([1]):
            angles[reject] = sample_angles(M, scales)
        else:
            angles[reject] = sample_angles(M, scales[reject])
        return angles


def uniformball_unitquat_rand(q, scales):
    r"""Sample rotations at random in balls centered around q,
    with radii given by the scales array."""

    N, d = list(q.size())
    angles = sample_angles(N, scales).reshape(N, 1)
    # Direction, randomly sampled on the sphere
    directions = torch.randn(N, 3, dtype=scales.dtype, device=scales.device)
    directions = directions / torch.norm(directions, dim=1).reshape((N, 1))

    q0 = angles_directions_to_quat(angles, directions)  # Uniform around 1

    return quat_mult(q, q0)


def uniform_unitquat_rand(N, device, dtype=torch.float32):
    r"""Uniform sample in the space of unit quaternions.

    Args:
        N (int): Number of samples.
        device (torch.device): Device on which the samples are created.
        dtype (torch.dtype, optional): Default is torch.float32.

    Returns:
        (N,4) Tensor:

        Samples.

    """
    angles = sample_angles(N, torch.tensor([math.pi],
                                           dtype=dtype, device=device))
    directions = torch.randn(N, 3, dtype=dtype, device=device)
    directions = directions / torch.norm(directions, dim=1).reshape((N, 1))
    return angles_directions_to_quat(angles, directions)


""" Rejection sampler for the von Mises Fisher distribution 
on the space of unit quaternions using the BACG method.
"""

def sample_q0(kappa, N):
    r"""Sample **N** random variables distributed according to the von
    Mises distribution on the space of quaternions with center
    :math:`(1,0,0,0)` and concentration parameter **kappa**.

    Use the BACG method described in [KGM2018]_.

    .. [KGM2018] J. Kent, A. Ganeiber & K. Mardia, A New Unified Approach
       for the Simulation of a Wide Class of Directional Distributions,
       *J. Comput. Graph. Statist.*, Vol. 27, No. 2 (2018).

    Args:
        kappa ((N,) Tensor or (1,) Tensor): Concentration parameter.
        N (int): Number of samples. Must be equal to the size of kappa
            when the size of kappa is >1.

    Returns:
        (N,) Tensor:

        Sample.

    """
    if N == 0:
        return torch.tensor([], device=kappa.device)
    ell = 2 * kappa
    b0 = torch.sqrt((ell - 2) ** 2 + 2 * ell) - (ell - 2)
    logM = -.5 * (4 - b0) + 2 * torch.log(4 / b0)
    if kappa.shape[0] == 1:
        A = torch.tensor([0., 2. * kappa, 2. * kappa, 2. * kappa],
                         dtype=kappa.dtype, device=kappa.device)
        Omega = torch.ones(4,
                           dtype=kappa.dtype,
                           device=kappa.device) + (2 / b0) * A
        Sigma = Omega ** (-1)
    else:
        A = torch.cat((
            torch.zeros(N, 1, dtype=kappa.dtype, device=kappa.device),
            (2. * kappa).reshape(N, 1),
            (2. * kappa).reshape(N, 1),
            (2. * kappa).reshape(N, 1)),
            dim=1)
        Omega = torch.ones(1, 4, dtype=kappa.dtype, device=kappa.device) \
                + (2. / b0.reshape(N, 1)) * A
        Sigma = Omega ** (-1)
    y = torch.sqrt(Sigma) * torch.randn(N, 4,
                                        dtype=kappa.dtype, device=kappa.device)
    q0 = y / torch.norm(y, dim=1).reshape((N, 1))
    U = torch.rand(N, dtype=kappa.dtype, device=kappa.device)
    u = torch.sum(A * q0 ** 2, dim=1)
    log_acg = -2 * torch.log(torch.sum(Omega * q0 ** 2, dim=1))
    reject = torch.log(U) > -u - logM - log_acg
    Nreject = reject.sum()
    if Nreject == 0:
        return q0
    else:
        if kappa.shape[0] == 1:
            q0[reject, :] = sample_q0(kappa, Nreject)
        else:
            q0[reject, :] = sample_q0(kappa[reject], Nreject)
        return q0


def vonmises_quat_rand(q, kappa):
    r""" Von Mises random variables on the space of quaternions with
    center **q** and concentration parameter **kappa**.

    Multiply **q** with a sample from the von Mises distribution on the
    space of quaternions with center :math:`(1,0,0,0)` and concentration
    parameter **kappa**. See :func:`sample_q0`.

    Args:
        q ((N,4) Tensor): Center of the distribution.
        kappa ((N,) Tensor or (1,) Tensor): Concentration parameter.

    Returns:
        (N,4) Tensor:

        Sample.

    """
    q0 = sample_q0(kappa, q.shape[0])
    return quat_mult(q, q0)

