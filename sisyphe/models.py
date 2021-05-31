import torch
import math
from pykeops.torch import LazyTensor
from .particles import Particles, KineticParticles, BOParticles, volume_ball, lazy_interaction_kernel, lazy_quadratic
from .sampling import vonmises_rand, vonmises_quat_rand, uniform_sphere_rand, uniform_angle_rand


class AsynchronousVicsek(KineticParticles):
    r"""Vicsek model with random jumps.

    The asynchronous Vicsek Model is described in [DM2016]_.

    The velocity of each particle is a unit vector in the sphere
    :math:`\mathbb{S}^{d-1}`. The velocity of each particle is updated
    at random times by computing first a *target* (Tensor) in
    :math:`\mathbb{R}^d` using the parameters contained in :attr:`target`
    and then a new velocity in :math:`\mathbb{S}^{d-1}` is
    sampled using a method specified by :attr:`sampling_method`. Between
    two jumps, a particle moves in straight line at a constant speed.

    .. [DM2016] G. Di Marco, S. Motsch, Self-alignment driven by jump
       processes: Macroscopic limit and numerical investigation,
       *Math. Models Methods Appl. Sci.* Vol. 26, No. 7 (2016).

    Attributes:
        name (str): ``"Asynchronous Vicsek"`` and the value of the key
            ``"name"`` in the dictionary :attr:`target`.
        v (float): The speed of the particles.
        dt (float): Discretisation time-step. It is equal to
            0.01 divided by :attr:`jumprate`.
        jumprate (float): Each particle has an independent Poisson clock
            with parameter :attr:`jumprate`.
        kappa ((1,) Tensor): Concentration parameter.
        target (dict): Target dictionary. Contains the target name and
            its parameters. To be passed to the method
            :meth:`compute_target` in the argument ``which_target``.
        options (dict): Options dictionary. Each key is an option's
            name and its value contains the parameters of the option.
            To be passed to the method :meth:`compute_target` in the
            argument ``which_options``.
        sampling_method (str): The new velocity is sampled using one of
            the following methods.

                * ``"vonmises"`` : Von Mises distribution with center
                  given by the normalised :attr:`target` and the
                  concentration parameter given by the norm of
                  :attr:`target`.

                * ``"uniform"`` : Uniform distribution in the ball
                  around the normalised :attr:`target` with deviation
                  given by the inverse of the norm of :attr:`target`.
                  Dimension 2 only.

                * ``"projected_normal"`` : Sample first a gaussian
                  random variable in :math:`\mathbb{R}^d` with center
                  the normalised :attr:`target` and standard deviation
                  the inverse of the norm of :attr:`target`. Then
                  project the result on the sphere
                  :math:`\mathbb{S}^{d-1}`.

                * ``"vectorial_noise"`` : Same as ``"projected_normal"``
                  where the normally distributed random variable is
                  replaced by a uniformly distributed random variable on
                  the sphere, divided by the norm of :attr:`target`.

        parameters (str): The parameters of the model.

    """

    def __init__(self, pos, vel,
                 v,
                 jump_rate, kappa,
                 interaction_radius,
                 box_size,
                 vision_angle=2 * math.pi,
                 axis=None,
                 boundary_conditions='periodic',
                 variant=None,
                 options=None,
                 sampling_method='vonmises',
                 block_sparse_reduction=False,
                 number_of_cells=1024):
        r"""

        Args:
            pos ((N,d) Tensor): Positions.
            vel ((N,d) Tensor): Velocities.
            v (float): The speed of the particles.
            jump_rate (float): Jump rate.
            kappa (float): Concentration parameter.
            interaction_radius (float or (N,) Tensor): Radius.
            box_size (float or list of size d): Box size.
            vision_angle (float, optional): Vision angle. Default is
                :math:`2\pi`.
            axis ((N,d) Tensor or None, optional): Axis of the
                particles. Default is None.
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
            variant (dict, optional): The variant of the model is
                defined by the target method. Default is::

                {"name" : "normalised", "parameters" : {}}.

            options (dict, optional): Dictionary of options to customise
                a target. Default is ``{}``.
            sampling_method (str, optional): Default is "vonmises".
            block_sparse_reduction (bool, optional): Use block sparse
                reduction or not. Default is False.
            number_of_cells (int, optional): Maximum number of cells if
                **block_sparse_reduction** is True. Will be rounded to
                the nearest lower power :attr:`d`. Default is 1024.

        """
        super().__init__(pos=pos, vel=vel,
                         interaction_radius=interaction_radius,
                         box_size=box_size,
                         vision_angle=vision_angle,
                         axis=axis,
                         boundary_conditions=boundary_conditions,
                         block_sparse_reduction=block_sparse_reduction,
                         number_of_cells=number_of_cells)
        self.N, self.d = list(pos.size())
        self.v = v
        self.dt = .01 / jump_rate
        self.jumprate = jump_rate
        self.kappa = torch.tensor([kappa],
                                  dtype=self.vel.dtype, device=self.vel.device)
        if variant is None:
            self.target = {"name": "normalised", "parameters": {}}
        else:
            self.target = variant
        self.options = {} if options is None else options
        self.sampling_method = sampling_method
        self.name = 'Asynchronous Vicsek ' + '(' + self.target["name"] + ')'
        self.parameters = 'N=' + str(self.N) \
                          + ' ; R=' + str(round(self.R, 2)) \
                          + ' ; Jump Rate=' + str(round(jump_rate, 2)) \
                          + ' ; kappa=' + str(round(kappa, 2)) \
                          + ' ; v=' + str(round(v, 2))

    def update(self):
        r"""Update the positions and velocities.

        Update the positions using a first order explicit Euler method
        during the time step :attr:`dt`.
        The particles which update their velocities in the time interval
        :attr:`dt` are given by the boolean mask of size :attr:`N`
        **who_jumps**. The probability for a particle to jump
        is equal to ``exp(-jumprate * dt)``.

        Returns:
            dict:

            Positions and velocities.

        """
        self.pos += self.v * self.dt * self.vel
        self.check_boundary()
        who_jumps = torch.rand(
            self.N,
            dtype=self.pos.dtype,
            device=self.pos.device) > math.exp(-self.jumprate * self.dt)
        M = who_jumps.sum()
        if M > 0:
            targets = self.compute_target(
                self.target,
                self.options, kappa=self.kappa, who=who_jumps)
            concentration = torch.norm(targets, dim=1)
            Omega = targets / concentration.reshape((M, 1))
            if self.sampling_method == 'vonmises':
                self.vel[who_jumps, :] = vonmises_rand(Omega, concentration)
            elif self.sampling_method == 'uniform':
                self.vel[who_jumps, :] = uniform_angle_rand(Omega,
                                                            1. / concentration)
            elif self.sampling_method == 'projected_normal':
                new_vel = Omega \
                          + torch.sqrt(1 / concentration.reshape((M, 1))) \
                          * self.vel.new(M,self.d).normal_()
                self.vel[who_jumps, :] = new_vel / torch.norm(
                    new_vel, dim=1).reshape((M, 1))
            elif self.sampling_method == 'vectorial_noise':
                xi = uniform_sphere_rand(M, self.d, targets.device,
                                         dtype=targets.dtype)
                vector = targets + xi
                self.vel[who_jumps, :] = vector / torch.norm(
                    vector, dim=1).reshape((M, 1))
            else:
                raise NotImplementedError()
        info = {
            "position": self.pos,
            "velocity": self.vel
        }
        return info


class Vicsek(KineticParticles):
    r"""Vicsek model with gradual alignment and diffusion.

    The Vicsek model is described in [DM2008]_.

    The velocity of each particle is a unit vector in the sphere
    :math:`\mathbb{S}^{d-1}``. The velocity of each particle is
    continuously updated by a combination of two effects.

        - A deterministic drift tends to relax the velocity towards a
          *target* computed using :attr:`target`. The intensity of the
          drift may depend on the norm of the target.

        - A random diffusion.

    The speed of the particles is constant.

    .. [DM2008] P. Degond, S. Motsch, Continuum limit of self-driven
       particles with orientation interaction, *Math. Models Methods
       Appl. Sci.* Vol. 18, Suppl. (2008).

    Attributes:
        name (str): ``"Vicsek"`` and the value of the key ``"name"`` in
            the dictionary :attr:`target`.
        v (float): The speed of the particles.
        dt (float): Discretisation time-step.
        sigma (float): The diffusion coefficient.
        nu (float): The drift coefficient
        kappa ((1,) Tensor): Concentration parameter equal to the ratio
            of :attr:`nu` over :attr:`sigma`.
        target (dict): Target dictionary. Contains the target name and
            its parameters. To be passed to the method
            :meth:`compute_target` in the argument ``which_target``.
        options (dict): Options dictionary. Each key is an option's
            name and its value contains the parameters of the option.
            To be passed to the method :meth:`compute_target` in the
            argument ``which_options``.
        scheme (str): The numerical scheme used to discretise the SDE.
            Can be one of the following:

                - ``"projection"`` : Projected Euler-Maruyama scheme
                  on the manifold :math:`\mathbb{S}^{d-1}`.

                - ``"expEM"`` : In dimension 2, Euler-Maruyama scheme in
                  the Lie algebra and exponentiation.

        parameters (str): The parameters of the model.

    """

    def __init__(self, pos, vel,
                 v,
                 sigma, nu,
                 interaction_radius,
                 box_size,
                 vision_angle=2 * math.pi,
                 axis=None,
                 boundary_conditions='periodic',
                 variant=None,
                 options=None,
                 numerical_scheme='projection',
                 dt=.01,
                 block_sparse_reduction=False,
                 number_of_cells=1600):
        r"""

        Args:
            pos ((N,d) Tensor): Positions.
            vel ((N,d) Tensor): Velocities.
            v (float): The speed of the particles.
            sigma (float): Diffusion parameter.
            nu (float): Drift parameter.
            interaction_radius (float or (N,) Tensor): Radius.
            box_size (float or list of size d): Box size.
            vision_angle (float, optional): Vision angle. Default is
                :math:`2\pi`.
            axis ((N,d) Tensor or None, optional): Axis of the
                particles. Default is None.
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
            variant (dict, optional): The variant of the model is
                defined by the target method. Default is::

                {"name" : "normalised", "parameters" : {}}.

            options (dict, optional): Dictionary of options to customise
                a target. Default is ``{}``.
            numerical_scheme (str, optional): The numerical scheme.
                Default is ``"projection"``.
            dt (float, optional): Discretisation time step. Default is
                0.01.
            block_sparse_reduction (bool, optional): Use block sparse
                reduction or not. Default is False.
            number_of_cells (int, optional): Maximum number of cells if
                **block_sparse_reduction** is True. Will be rounded to
                the nearest lower power :attr:`d`. Default is 1024.

        """
        super().__init__(pos=pos, vel=vel,
                         interaction_radius=interaction_radius,
                         box_size=box_size,
                         vision_angle=vision_angle,
                         axis=axis,
                         boundary_conditions=boundary_conditions,
                         block_sparse_reduction=block_sparse_reduction,
                         number_of_cells=number_of_cells)
        self.N, self.d = list(pos.size())
        self.v = v
        self.scheme = numerical_scheme
        self.dt = dt
        self.sigma = sigma
        self.nu = nu
        self.kappa = nu / sigma
        if variant is None:
            self.target = {"name": "normalised", "parameters": {}}
        else:
            self.target = variant
        self.options = {} if options is None else options
        self.name = 'Vicsek' + '(' + self.target["name"] + ')'
        self.parameters = 'N=' + str(self.N) \
                          + ' ; R=' + str(round(self.R, 2)) \
                          + ' ; nu=' + str(round(self.nu, 2)) \
                          + ' ; sigma=' + str(round(self.sigma, 2)) \
                          + ' ; v=' + str(round(self.v, 2))

    def update(self):
        r"""Update the positions and velocities.

        Update the positions using a first order explicit Euler method
        during the time step :attr:`dt`.
        Update the velocities using the numerical scheme
        :attr:`scheme` and the method :meth:`one_step_velocity_scheme`.

        Returns:
            dict:

            positions and velocities.

        """
        self.pos += self.v * self.dt * self.vel
        self.check_boundary()
        targets = self.compute_target(self.target, self.options,
                                      kappa=self.kappa)
        self.vel = self.one_step_velocity_scheme(targets)
        info = {
            "position": self.pos,
            "velocity": self.vel
        }
        return info

    def one_step_velocity_scheme(self, targets):
        r"""One time step of the numerical scheme for the velocity.

        Args:
            targets ((N,d) Tensor)

        Returns:
            (N,d) Tensor

        """
        if self.scheme == 'projection':
            dB = self.vel.new(self.N, self.d).normal_()
            inRd = self.sigma * targets * self.dt \
                   + math.sqrt(2 * self.sigma * self.dt) * dB
            orth = torch.einsum('ij,ij->i', self.vel, inRd)
            proj = inRd - orth.reshape(self.N, 1) * self.vel
            # Stratonovich correction
            dv = proj - self.sigma * (self.d - 1) * self.vel * self.dt
            new_vel = self.vel + dv
            new_vel = new_vel / torch.norm(new_vel, dim=1).reshape((self.N, 1))

            # # Check if there are some nan....
            # notcool = torch.isnan(new_vel[:, 0])
            # nb_notcool = int(notcool.sum())
            # if nb_notcool > 0:
            #     self.vel[notcool, :] = uniform_sphere_rand(
            #         nb_notcool, self.d, self.vel.device, dtype=self.vel.dtype)

        elif self.scheme == 'expEM':
            theta = torch.atan2(self.vel[:, 1], self.vel[:, 0])
            theta = theta.reshape((self.N, 1))
            dB = self.vel.new(self.N, self.d).normal_()
            inRd = self.sigma * targets * self.dt \
                   + math.sqrt(2 * self.sigma * self.dt) * dB
            v_orth = torch.cat(
                (-self.vel[:, 1].reshape((self.N, 1)),
                 self.vel[:, 0].reshape((self.N, 1))),
                dim=1)
            scalar_orth = torch.einsum('ij,ij->i', v_orth, inRd)
            theta = theta + scalar_orth.reshape((self.N, 1))
            new_vel = torch.cat(
                (torch.cos(theta),
                 torch.sin(theta)),
                dim=1).reshape((self.N, 2))

        #         elif self.scheme=='circle':
        #             if not self.normalize:
        #                 raise NotImplementedError
        #             else:
        #                 theta = torch.atan2(self.vel[:,1],self.vel[:,0]).type_as(self.vel)
        #                 B = self.vel+.5*(targets-self.vel)*self.nu*self.dt
        #                 angle = torch.atan2(B[:,1],B[:,0])-theta
        #                 new_angle = theta + 2*angle+math.sqrt(2*self.sigma*self.dt)*torch.randn(self.N).type_as(self.vel)
        #                 new_angle = new_angle.reshape((self.N,1))
        #                 new_vel = torch.cat((torch.cos(new_angle),torch.sin(new_angle)),dim=1).reshape((self.N,2))
        else:
            raise NotImplementedError()
        return new_vel


class SynchronousVicsek(KineticParticles):
    r"""Vicsek model with synchronous random jumps.

    This is the original Vicsek model described in [CGGR2008]_.

    The velocity of each particle is a unit vector in the sphere
    :math:`\mathbb{S}^{d-1}`. The velocity of each particle is updated
    at each iteration by computing first a *target* (Tensor) in
    :math:`\mathbb{R}^d` using the parameters contained in the
    dictionary :attr:`target` and then a new velocity in
    :math:`\mathbb{S}^{d-1}` is sampled using a method specified by
    :attr:`sampling_method`. The particles move in straight line at a
    constant speed.

    .. [CGGR2008] H. Chaté, F. Ginelli, G. Grégoire, F. Raynaud,
       Collective motion of self-propelled particles interacting without
       cohesion, *Phys. Rev. E.*, Vol. 77, No. 4 (2008).

    Attributes:
        name (str): ``"Synchronous Vicsek"`` and the value of the key
            ``"name"`` in the dictionary :attr:`target`.
        v (float): The speed of the particles.
        dt (float): Discretisation time-step.
        kappa ((1,) Tensor): Concentration parameter.
        target (dict): Target dictionary. Contains the target name and
            its parameters. To be passed to the method
            :meth:`compute_target` in the argument ``which_target``.
        options (dict): Options dictionary. Each key is an option's
            name and its value contains the parameters of the option.
            To be passed to the method :meth:`compute_target` in the
            argument ``which_options``.
        sampling_method (str): The new velocity is sampled using one of
            the following methods.

                * ``"vonmises"`` : Von Mises distribution with center
                  given by the normalised :attr:`target` and the
                  concentration parameter given by the norm of
                  :attr:`target`.

                * ``"uniform"`` : Uniform distribution in the ball
                  around the normalised :attr:`target` with deviation
                  given by the inverse of the norm of :attr:`target`.
                  Dimension 2 only.

                * ``"projected_normal"`` : Sample first a gaussian
                  random variable in :math:`\mathbb{R}^d` with center
                  the normalised :attr:`target` and standard deviation
                  the inverse of the norm of :attr:`target`. Then
                  project the result on the sphere
                  :math:`\mathbb{S}^{d-1}`.

                * ``"vectorial_noise"`` : Same as ``"projected_normal"``
                  where the normally distributed random variable is
                  replaced by a uniformly distributed random variable on
                  the sphere, divided by the norm of :attr:`target`.
        parameters (str): The parameters of the model.

    """

    def __init__(self, pos, vel,
                 v,
                 dt,
                 kappa,
                 interaction_radius,
                 box_size,
                 vision_angle=2 * math.pi,
                 axis=None,
                 boundary_conditions='periodic',
                 variant=None,
                 options=None,
                 sampling_method='uniform',
                 block_sparse_reduction=False,
                 number_of_cells=1024):
        r"""

        Args:
            pos ((N,d) Tensor): Positions.
            vel ((N,d) Tensor): Velocities.
            v (float): The speed of the particles.
            kappa (float): Concentration parameter.
            interaction_radius (float or (N,) Tensor): Radius.
            box_size (float or list of size d): Box size.
            vision_angle (float, optional): Vision angle. Default is
                :math:`2\pi`.
            axis ((N,d) Tensor or None, optional): Axis of the
                particles. Default is None.
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
            variant (dict, optional): The variant of the model is
                defined by the target method. Default is::

                {"name" : "normalised", "parameters" : {}}.

            options (dict, optional): Dictionary of options to customise
                a target. Default is ``{}``.
            sampling_method (str, optional): Default is "vonmises".
            block_sparse_reduction (bool, optional): Use block sparse
                reduction or not. Default is False.
            number_of_cells (int, optional): Maximum number of cells if
                **block_sparse_reduction** is True. Will be rounded to
                the nearest lower power :attr:`d`. Default is 1024.

        """
        super().__init__(pos=pos, vel=vel,
                         interaction_radius=interaction_radius,
                         box_size=box_size,
                         vision_angle=vision_angle,
                         axis=axis,
                         boundary_conditions=boundary_conditions,
                         block_sparse_reduction=block_sparse_reduction,
                         number_of_cells=number_of_cells)
        self.N, self.d = list(pos.size())
        self.v = v
        self.dt = dt
        self.kappa = torch.tensor([kappa],
                                  dtype=self.vel.dtype, device=self.vel.device)
        if variant is None:
            self.target = {"name": "normalised", "parameters": {}}
        else:
            self.target = variant
        self.options = {} if options is None else options
        self.sampling_method = sampling_method
        self.name = 'Synchronous Vicsek ' + '(' + self.target["name"] + ')'
        self.parameters = 'N=' + str(self.N) \
                          + ' ; R=' + str(self.R) \
                          + ' ; dt=' + str(dt) \
                          + ' ; kappa=' + str(kappa) \
                          + ' ; v=' + str(v)

    def update(self):
        r"""Update the positions and velocities.

        Update the positions using a first order explicit Euler method
        during the time step :attr:`dt`.
        Update the velocities by sampling new velocities using the
        method given by :attr:`sampling_method`.

        Returns:
            dict:

            Positions and velocities.

        """
        self.pos += self.v * self.dt * self.vel
        self.check_boundary()
        targets = self.compute_target(self.target, self.options,
                                      kappa=self.kappa)
        concentration = torch.norm(targets, dim=1)
        Omega = targets / concentration.reshape((self.N, 1))
        if self.sampling_method == 'vonmises':
            self.vel = vonmises_rand(Omega, concentration)
        elif self.sampling_method == 'uniform':
            self.vel = uniform_angle_rand(Omega, 1. / concentration)
        elif self.sampling_method == 'projected_normal':
            new_vel = Omega \
                      + torch.sqrt(1 / concentration.reshape((self.N, 1))) \
                      * self.vel.new(self.N, self.d).normal_()
            self.vel = new_vel / torch.norm(
                new_vel, dim=1).reshape((self.N, 1))
        elif self.sampling_method == 'vectorial_noise':
            xi = uniform_sphere_rand(self.N, self.d, targets.device,
                                     dtype=targets.dtype)
            vector = targets + xi
            self.vel = vector / torch.norm(vector, dim=1).reshape((self.N, 1))
        else:
            raise NotImplementedError()

        info = {
            "position": self.pos,
            "velocity": self.vel
        }
        return info


class CuckerSmale(KineticParticles):

    def __init__(self, pos, vel,
                 sigma, nu,
                 interaction_radius,
                 box_size,
                 alpha,
                 beta,
                 vision_angle=2 * math.pi,
                 axis=None,
                 boundary_conditions='periodic',
                 variant=None,
                 options=None,
                 dt=.01,
                 block_sparse_reduction=False,
                 number_of_cells=1600):
        super().__init__(pos=pos, vel=vel,
                         interaction_radius=interaction_radius,
                         box_size=box_size,
                         vision_angle=vision_angle,
                         axis=axis,
                         boundary_conditions=boundary_conditions,
                         block_sparse_reduction=block_sparse_reduction,
                         number_of_cells=number_of_cells)
        self.name = 'Cucker-Smale'
        self.N, self.d = list(pos.size())
        self.dt = dt
        self.sigma = sigma
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.target = variant
        self.options = options
        self.parameters = 'N=' + str(self.N) \
                          + ' ; R=' + str(round(self.R, 2)) \
                          + ' ; nu=' + str(round(nu, 2)) \
                          + ' ; sigma=' + str(round(sigma, 2)) \
                          + ' ; L=' + str(round(box_size, 2))

    def update(self):
        self.pos += self.dt * self.vel
        self.check_boundary()
        justN = torch.ones((self.N, 1),
                           dtype=self.pos.dtype, device=self.pos.device)
        J, N_neighbours = self.linear_local_average(self.vel, justN)
        scaling = (self.L / self.R) ** self.d * 1. / volume_ball(self.d)
        self_prop_force = \
            self.alpha * self.vel \
            - self.beta * ((self.vel ** 2).sum(axis=1).reshape((self.N, 1))) \
            * self.vel
        self.vel += \
            self.dt * self_prop_force \
            + self.nu * self.dt * scaling * (J - N_neighbours * self.vel) \
            + math.sqrt(2 * self.sigma * self.dt) \
            * self.vel.new(self.N, self.d).normal_()
        info = {"position": self.pos, "velocity": self.vel}
        return info


class BOAsynchronousVicsek(BOParticles):
    r"""Asynchronous Vicsek model for body-oriented particles.

    The asynchronous Vicsek Model is described in [DFM2017]_.

    The body-orientation of each particle is updated at random times
    by computing first a *target* (Tensor) in :math:`SO(3)` using the
    parameters contained in the dictionary :attr:`target` and then a
    new body-orientation in :math:`SO(3)` is sampled using a method
    specified by :attr:`sampling_method`. Between two jumps, a particle
    moves in straight line at a constant speed.

    .. [DFM2017] P. Degond, A. Frouvelle, S. Merino-Aceituno, A new
       flocking model through body attitude coordination, *Math. Models
       Methods Appl. Sci.* Vol. 27, No. 6 (2017).


    Attributes:
        name (str): ``"Body-Orientation Asynchronous Vicsek"`` and the
            value of the key ``"name"`` in the dictionary :attr:`target`.
        v (float): The speed of the particles.
        dt (float): Discretisation time-step. It is equal to
            0.01 divided by :attr:`jumprate`.
        jumprate (float): Each particle has an independent Poisson clock
            with parameter :attr:`jumprate`.
        kappa ((1,) Tensor): Concentration parameter.
        target (dict): Target dictionary. Contains the target name and
            its parameters. To be passed to the method
            :meth:`compute_target` in the argument ``which_target``.
        options (dict): Options dictionary. Each key is an option's
            name and its value contains the parameters of the option.
            To be passed to the method :meth:`compute_target` in the
            argument ``which_options``.
        sampling_method (str): The new velocity is sampled using one of
            the following methods.

                * ``"vonmises"`` : Von Mises distribution with center
                  given by the normalised :attr:`target` and the
                  concentration parameter given by the norm of
                  :attr:`target`.

                * ``"uniform"`` : Uniform distribution in the ball
                  around the normalised :attr:`target` with deviation
                  given by the inverse of the norm of :attr:`target`.
                  Dimension 2 only.

        parameters (str): The parameters of the model.

    """

    def __init__(self, pos, bo,
                 v,
                 jump_rate, kappa,
                 interaction_radius,
                 box_size,
                 vision_angle=2 * math.pi,
                 axis=None,
                 boundary_conditions='periodic',
                 variant=None,
                 options=None,
                 sampling_method='vonmises',
                 block_sparse_reduction=False,
                 number_of_cells=1600):
        r"""

        Args:
            pos ((N,3) Tensor): Positions.
            bo ((N,4) Tensor): Body-orientation as unit quaternions.
            v (float): The speed of the particles.
            jump_rate (float): Jump rate.
            kappa (float): Concentration parameter.
            interaction_radius (float or (N,) Tensor): Radius.
            box_size (float or list of size d): Box size.
            vision_angle (float, optional): Vision angle. Default is
                :math:`2\pi`.
            axis ((N,3) Tensor or None, optional): Axis of the
                particles. Default is None.
            boundary_conditions (list or str, optional): Boundary conditions.
                Can be one of the following:

                - list of size :math:`d` containing 0 (periodic) or 1
                  (wall with reflecting boundary conditions) for each
                  dimension.

                - ``"open"`` : no boundary conditions.

                - ``"periodic"`` : periodic boundary conditions.

                Default is ``"periodic"``.
            variant (dict, optional): The variant of the model is
                defined by the target method. Default is::

                {"name" : "normalised", "parameters" : {}}.

            options (dict, optional): Dictionary of options to customise
                a target. Default is ``{}``.
            sampling_method (str, optional): Default is ``"vonmises"``.
            block_sparse_reduction (bool, optional): Use block sparse
                reduction or not. Default is False.
            number_of_cells (int, optional): Maximum number of cells if
                **block_sparse_reduction** is True. Will be rounded to
                the nearest lower power :attr:`d`. Default is 1024.

        """
        super().__init__(pos=pos, bo=bo,
                         interaction_radius=interaction_radius,
                         box_size=box_size,
                         vision_angle=vision_angle,
                         axis=axis,
                         boundary_conditions=boundary_conditions,
                         block_sparse_reduction=block_sparse_reduction,
                         number_of_cells=number_of_cells)
        self.name = 'Body-Orientation Asynchronous Vicsek'
        self.N, self.d = list(pos.size())
        self.v = v
        self.dt = .01 / jump_rate
        self.jumprate = jump_rate
        self.kappa = torch.tensor([kappa],
                                  dtype=self.bo.dtype, device=self.bo.device)
        if variant is None:
            self.target = {"name": "normalised", "parameters": {}}
        else:
            self.target = variant
        self.options = {} if options is None else options
        self.sampling_method = sampling_method
        self.parameters = 'N=' + str(self.N) \
                          + ' ; R=' + str(self.R) \
                          + ' ; Jump Rate=' + str(jump_rate) \
                          + ' ; kappa=' + str(kappa) \
                          + ' ; v=' + str(v)

    def update(self):
        r"""Update the positions and body-orientations.

        Update the positions using a first order explicit Euler method
        during the time step :attr:`dt`.
        The particles which update their body-orientations in the time
        interval :attr:`dt` are given by the boolean mask of size
        :attr:`N` **who_jumps**. The probability for a particle
        to jump is equal to ``exp(-jumprate * dt)``.

        Returns:
            dict:

            Positions and body-orientations.

        """
        self.pos += self.v * self.dt * self.vel
        self.check_boundary()
        who_jumps = torch.rand(
            self.N,
            dtype=self.pos.dtype,
            device=self.pos.device) > math.exp(-self.jumprate * self.dt)
        M = who_jumps.sum()
        if M > 0:
            targets = self.compute_target(self.target, self.options, who=who_jumps)
            concentration = self.kappa
            if self.sampling_method == 'uniform':
                # self.bo[who_jumps, :] = uniformball_unitquat_rand(
                #     targets, 1 / concentration)
                raise Exception('bug')
            elif self.sampling_method == 'vonmises':
                self.bo[who_jumps, :] = vonmises_quat_rand(targets,
                                                           concentration)
            else:
                raise NotImplementedError()
        info = {
            "position": self.pos,
            "body-orientation": self.bo
        }
        return info


class VolumeExclusion(Particles):
    r"""Spherical particles interacting via short-range repulsion.

    The model is described in [MP2017]_.

    Each particle has a fixed radius and the particles repulse each
    other when they overlap. In addition the particles die and can
    divide at a constant rate. Currently implemented in an open domain
    only.

    .. [MP2017] S. Motsch, D. Peurichard, From short-range repulsion to
       Hele-Shaw problem in a model of tumor growth, *J. Math. Biology*,
       Vol. 76, No. 1, 2017.


    Attributes:
        alpha (float): Drift coefficient.
        mu (float): Division rate.
        nu (float): Death rate.
        dt0 (float): Default time step.
        dt (float): Current discretisation time step (adaptive).
        Nmax (int): Maximum number of particles.
        age ((self.N,) Tensor): Age of the particles.
        name (str): ``"Volume exclusion"``.

    """

    def __init__(self,
                 pos,
                 interaction_radius,
                 box_size,
                 alpha,
                 division_rate,
                 death_rate,
                 dt,
                 Nmax=20000,
                 boundary_conditions="open",
                 block_sparse_reduction=False,
                 number_of_cells=1024):
        r"""

        Args:
            pos ((N,d) Tensor): Positions.
            interaction_radius (float or (N,) Tensor): Radius.
            box_size (float or list of size d): Box size.
            alpha (float): Drift coefficient.
            division_rate (float): Division rate.
            death_rate (float): Death rate.
            dt (float): Default time step.
            Nmax (int, optional): Maximal number of particles. Default
                is 20000.
            boundary_conditions (str, optional): Default is "open".
            block_sparse_reduction (bool, optional): Use block sparse
                reduction or not. Default is False.
            number_of_cells (int, optional): Maximum number of cells if
                **block_sparse_reduction** is True. Will be rounded to
                the nearest lower power :attr:`d`. Default is 1024.

        """

        super().__init__(pos=pos,
                         interaction_radius=interaction_radius,
                         box_size=box_size,
                         vision_angle=2 * math.pi,
                         axis=None,
                         boundary_conditions=boundary_conditions,
                         block_sparse_reduction=block_sparse_reduction,
                         number_of_cells=number_of_cells)
        if isinstance(interaction_radius, float):
            self.R = torch.repeat_interleave(
                torch.tensor(interaction_radius,
                             device=self.pos.device, dtype=self.pos.dtype),
                self.N)
        if not self.bc == "open":
            raise ValueError("Currently implemented in an open domain only.")
        self.alpha = alpha
        self.mu = division_rate
        self.nu = death_rate
        self.dt0 = dt
        self.dt = dt
        self.Nmax = Nmax
        self.age = torch.zeros(self.N,
                               dtype=self.pos.dtype, device=self.pos.device)
        self.name = 'Volume exclusion'

    @property
    def parameters(self):
        return 'N=' + str(self.N) + ' ; ' + 'alpha=' + str(self.alpha) \
               + ' ; ' + 'Division rate=' + str(self.mu) \
               + ' ; ' + 'Death rate=' + str(self.nu)

    @property
    def E(self):
        r"""The energy of the particle system.

        The force exerted by a particle located at :math:`x_j` with
        radius :math:`R_j` on a particle located at :math:`x_i` with
        radius :math:`R_i` is

        .. math::

            F = -\frac{\alpha}{R_i} \nabla_{x_i} U\left(\frac{|x_i - x_j|^2}{(R_i + R_j)^2}\right),

        where the potential is

        .. math::

            U(s) = -\log(s) + s - 1\,\,\text{for}\,\, s<1 \,\,\text{and}\,\, U(s) = 0\,\, \text{for}\,\, s>1.

        The energy of the couple is

        .. math::

            E_{ij} = U\left(\frac{|x_i - x_j|^2}{(R_i + R_j)^2}\right)  (R_i + R_j)^2.

        The total energy of the system is

        .. math::

            E = -\frac{1}{2}\sum_{i\ne j} E_{ij}.

        Returns:
            (1,) Tensor:

            The total energy of the system.

        """
        x = self.pos  # (M,D)
        y = self.pos  # (N,D)
        Rx = self.R
        Ry = self.R
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(y[None, :, :])
        if type(Rx) == type(x):
            R_i = LazyTensor(Rx.reshape((x.shape[0], 1))[:, None])
            R_j = LazyTensor(Ry.reshape((y.shape[0], 1))[None, :])
            R_ij = R_i + R_j  # (M,N)
        else:
            R_ij = Rx + Ry  # float
        z_ij = x_i - y_j  # (M,N,D)
        sq_dist = (z_ij ** 2).sum(-1)  # (M,N)
        is_zero = -(sq_dist.sign() - 1)
        sq_xoverR = sq_dist / (R_ij ** 2) + is_zero
        E_ij = ((-sq_xoverR + sq_xoverR.log() + 1.)
                * (R_ij ** 2) * ((R_ij ** 2) / (sq_dist + is_zero) - 1).relu())
        return -.5 * E_ij.sum(axis=1).sum()

    def add_particle(self, who):
        r"""Some particle divide.

        When a particle at :math:`x_i` with radius :math:`R_i` divides,
        a new particle is added at position :math:`x_i + 2R_iu` where
        :math:`u` is uniformly sampled on the unit sphere. The radius of
        the new particle is :math:`R_i`.

        Args:
            who ((N,) BoolTensor): Boolean mask giving the particles
                which divide.

        """
        M = who.sum()
        u = self.pos.new(M,self.d).normal_()
        u = u / torch.norm(u, dim=1).reshape((M, 1))
        where = 2 * self.R[who].reshape((M, 1)) * u
        new_pos = self.pos[who, :] + where
        self.pos = torch.cat((self.pos, new_pos), dim=0)
        self.R = torch.cat((self.R, self.R[who]))
        self.age = torch.cat((self.age, torch.zeros(M,
                                                    dtype=self.pos.dtype,
                                                    device=self.pos.device)))
        self.N = self.pos.shape[0]

    def remove_particle(self, who):
        """The particles in the boolean mask **who** die. """
        self.R[who] = torch.zeros(who.sum(),
                                  dtype=self.R.dtype, device=self.R.device)
        self.pos = self.pos[~who, :]
        self.R = self.R[~who]
        self.age = self.age[~who]
        self.N = self.pos.shape[0]

    def update(self):
        r"""Update the positions.

        First compute the force. The Energy satisfies

        .. math::

            \frac{\mathrm{d}}{\mathrm{d}t} E = -\frac{1}{2} \sum_{i\ne j} \frac{\alpha}{R_i} \left|\frac{\mathrm{d}x_i}{\mathrm{d}t}\right|^2 \leq0.

        Update the position using an adaptive time step:

        1. Compute the energy of the system :attr:`E`.
        2. Set ``dt = dt0``.
        3. Compute the new positions with time step :attr:`dt`.
        4. Compute the energy of the new system.
        5. If the energy of the new system is higher than :attr:`E` then
           set ``dt = dt/2`` and return to 3.

        Once the positions are updated, compute the set of particles
        which divide during the time interval :attr:`dt`. Each particle
        has a probability ``exp(-dt * mu)`` to divide. Divisions
        are not allowed if there are more than :attr:`Nmax`
        particles.
        Finally compute the set of particles which die during the
        interval :attr:`dt`. Each particle has a probability
        ``exp(-dt * nu)`` to die. Death is not allowed if there
        are less that 2 particles.

        Returns:
            dict:

            Positions.

        """
        F = self.overlapping_repulsion_target()
        self.dt = min(self.dt0, float(.1 / F.max()))
        x = self.pos
        E = self.E
        self.pos = x + self.alpha / self.R.reshape((self.N, 1)) * self.dt * F
        while self.E > E:
            self.dt = self.dt / 2
            self.pos = x + self.alpha / self.R.reshape((self.N, 1)) * self.dt * F
            if self.dt < .00000000001:
                raise ValueError('This is too small!')
        who_divides = torch.rand(
            self.N,
            dtype=self.pos.dtype,
            device=self.pos.device) > math.exp(-self.mu * self.dt)
        if self.N < self.Nmax:
            self.add_particle(who_divides)
        who_dies = torch.rand(
            self.N,
            dtype=self.pos.dtype,
            device=self.pos.device) > math.exp(-self.nu * self.dt)
        if self.N > 2:
            self.remove_particle(who_dies)
        self.age = self.age + self.dt
        info = {"position": self.pos}
        return info


class AttractionRepulsion(KineticParticles):
    r"""Self-propelled particles with attraction and repulsion forces.

    The model is studied in [DCBC2006]_.

    The attraction-repulsion force is derived from the Morse potential.
    The self-propulsion force tends to relax the speed of the particles
    towards the ratio of the parameters :attr:`alpha` and
    :attr:`beta`.

    .. [DCBC2006] M. R. D’Orsogna, Y. L. Chuang, A. L. Bertozzi,
       L. S. Chayes, Self-Propelled Particles with Soft-Core
       Interactions: Patterns, Stability, and Collapse, *Phys. Rev.
       Lett.*, Vol. 96, No. 10 (2006).

    Attributes:
         alpha (float): Propulsion parameter.
         beta (float): Friction parameter.
         Ca (float): Attraction coefficient.
         la (float): Attraction length.
         Cr (float): Repulsion coefficient.
         lr (float): Repulsion length.
         p (int): Exponent of the Morse potential.
         dt (float): Time step.
         isaverage (bool): Use the mean-field scaling or not.
         mass ((N,) Tensor): The mass is proportional to the volume.
         name (str): ``'Self-propulsion and Attraction-Repulsion'``.
         parameters (str): Parameters.

    """

    def __init__(self,
                 pos,
                 vel,
                 interaction_radius,
                 box_size,
                 propulsion,
                 friction,
                 Ca,
                 la,
                 Cr,
                 lr,
                 dt,
                 p=2,
                 isaverage=False,
                 vision_angle=2 * math.pi,
                 axis=None,
                 boundary_conditions='open',
                 block_sparse_reduction=False,
                 number_of_cells=1000):
        r"""

        Args:
            pos ((N,d) Tensor): Positions.
            vel ((N,d) Tensor): Velocities.
            interaction_radius (float or (N,) Tensor): Radius.
            box_size (float or list of size d): Box size.
            propulsion (float): Propulsion parameter.
            friction (float): Friction parameter
            Ca (float): Attraction coefficient.
            la (float): Attraction length.
            Cr (float): Repulsion coefficient.
            lr (float): Repulsion length.
            dt (float): Time step.
            p (int, optiona,): Exponent of the Morse potential. Default
                is 2.
            isaverage (bool, optional): Use the mean-field scaling or
                not. Default is False.
            vision_angle (float, optional): Vision angle. Default is
                :math:`2\pi`.
            axis ((N,d) Tensor or None, optional): Axis of the
                particles. Default is None.
            boundary_conditions (str, optional): Default is ``"open"``.
            block_sparse_reduction (bool, optional): Use block sparse
                reduction or not. Default is False.
            number_of_cells (int, optional): Maximum number of cells if
                **block_sparse_reduction** is True. Will be rounded to
                the nearest lower power :attr:`d`. Default is 1024.
        """
        super().__init__(pos=pos, vel=vel,
                         interaction_radius=interaction_radius,
                         box_size=box_size,
                         vision_angle=vision_angle,
                         axis=axis,
                         boundary_conditions=boundary_conditions,
                         block_sparse_reduction=block_sparse_reduction,
                         number_of_cells=number_of_cells)
        if not self.bc == "open":
            raise ValueError("Currently implemented in an open domain only.")
        self.alpha = propulsion
        self.beta = friction
        self.Ca = Ca
        self.la = la
        self.Cr = Cr
        self.lr = lr
        self.dt = dt
        self.p = p
        self.isaverage = isaverage
        self.mass = self.R ** self.d
        if isinstance(self.mass, float):
            self.mass = torch.repeat_interleave(
                torch.tensor([self.mass],
                             dtype=self.pos.dtype, device=self.pos.device),
                self.N)
        self.name = 'Self-propulsion and Attraction-Repulsion'
        self.parameters = 'N=' + str(self.N) + ' ; ' \
                          + 'alpha=' + str(self.alpha) + ' ; ' \
                          + 'beta=' + str(self.beta) + ' ; ' \
                          + 'Ca=' + str(self.Ca) + ' ; ' \
                          + 'la=' + str(self.la) + ' ; ' \
                          + 'Cr=' + str(self.Cr) + ' ; ' \
                          + 'lr=' + str(self.lr)

    def compute_force(self):
        r"""Return the sum of the self-propulsion and
           attraction-repulsion forces. """
        v = self.vel
        F_prop = (
                self.alpha * v
                - self.beta * ((v ** 2).sum(axis=1).reshape((self.N, 1))) * v)
        F_ar = (1 / self.mass.reshape((self.N, 1))) * self.morse_target(
            Ca=self.Ca, la=self.la,
            Cr=self.Cr, lr=self.lr,
            p=self.p, mass=self.mass,
            isaverage=self.isaverage)
        return F_prop + F_ar

    def update(self):
        r"""Update the positions and velocities. RK4 numerical scheme.

        Returns:
            dict:

            Positions and velocities.

        """
        self.check_boundary()
        ### RK4 ###
        x = self.pos
        v = self.vel
        # Step 1
        k1x = self.vel
        k1v = self.compute_force()
        # Step 2
        self.pos = x + self.dt / 2 * k1x
        self.vel = v + self.dt / 2 * k1v
        k2x = self.vel
        k2v = self.compute_force()
        # Step 3
        self.pos = x + self.dt / 2 * k2x
        self.vel = v + self.dt / 2 * k2v
        k3x = self.vel
        k3v = self.compute_force()
        # Step 4
        self.pos = x + self.dt * k3x
        self.vel = v + self.dt * k3v
        k4x = self.vel
        k4v = self.compute_force()
        # Conclusion
        self.pos = x + (self.dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        self.vel = v + (self.dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)

        info = {"position": self.pos, "velocity": self.vel}
        return info