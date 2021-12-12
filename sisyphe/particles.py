import sys
import math
import copy
import time
import numpy as np
import torch
from pykeops.torch import LazyTensor
from .sampling import quat_mult
from .kernels import squared_distances_tensor, lazy_interaction_kernel, \
    lazy_morse, lazy_quadratic
from .toolbox import uniform_grid_centroids, volume_ball, \
    block_sparse_reduction_parameters, maximal_eigenvector_dim2
# from pykeops.torch.cluster import cluster_ranges
from pykeops.torch.cluster import sort_clusters

# from pykeops.torch.cluster import from_matrix

"""Particles as Python iterators."""


class Particles:
    r"""The main class.

    Attributes:
        N (int): Number of particles.
        d (int): Dimension.
        pos ((N,d) Tensor): Positions of the particles.
        R (float or (N,) Tensor): Radius of the particles.
        L ((d,) Tensor): Box size.
        angle (float): Vision angle.
        axis_is_none (bool): True when an axis is specified.
        axis ((N,d) Tensor or None): Axis of the particles.
        bc (list or str): Boundary conditions. Can be
            one of the following:

            - list of size :math:`d` containing 0 (periodic) or 1
              (wall with reflecting boundary conditions) for each
              dimension.

            - ``"open"`` : no boundary conditions.

            - ``"periodic"`` : periodic boundary conditions.

            - ``"spherical"`` : reflecting boundary conditions on the
              sphere of radius ``L[0]/2`` and center **L**/2.

        target_method (dict): Dictionary of target methods (see the
            function :func:`compute_target`).
        target_option_method (dict): Dictionary of options for the
            targets (see the function :func:`compute_target`).
        blocksparse (bool): Use block sparse reduction or not.
        iteration (int): Iteration.
        centroids (Tensor): Centroids.
        eps (Tensor): Size of the cells.
        keep (BoolTensor): ``keep[i,j]`` is True when the cells ``i``
            and ``j`` are contiguous.

    """

    def __init__(self,
                 pos,
                 interaction_radius,
                 box_size,
                 vision_angle=2 * math.pi,
                 axis=None,
                 boundary_conditions='periodic',
                 block_sparse_reduction=False,
                 number_of_cells=1024):
        r"""Initialise the attributes of the class.

        Args:
            pos ((N,d) Tensor): Positions.
            interaction_radius (float or (N,) Tensor): Radius.
            box_size (float or list of size d): Box size.
            vision_angle (float, optional): Vision angle. Default is
                2*pi.
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
            block_sparse_reduction (bool, optional): Use block sparse
                reduction or not. Default is False.
            number_of_cells (int, optional): Maximum number of cells if
                **block_sparse_reduction** is True. Will be rounded to
                the nearest lower power :attr:`d`. Default is 1024.

        """
        self.N, self.d = list(pos.size())
        self.pos = pos
        self.R = interaction_radius
        self.L = torch.tensor(box_size, dtype=pos.dtype, device=pos.device)
        if self.L.shape == torch.Size([]):
            self.L = torch.repeat_interleave(self.L, self.d)
        self.angle = vision_angle
        self.axis_is_none = axis is None
        self.axis = axis
        self.bc = boundary_conditions
        self.target_method = {}
        self.add_target_method("morse", self.morse_target)
        self.add_target_method("quadratic_potential",
                               self.quadratic_potential_target)
        self.add_target_method("overlapping_repulsion",
                               self.overlapping_repulsion_target)
        self.target_option_method = {}
        self.blocksparse = block_sparse_reduction
        (self.eps,
         self.keep,
         self.centroids) = self.compute_blocksparse_parameters(number_of_cells)
        self.iteration = 0

    def __iter__(self):
        return self

    def __next__(self):
        info = self.update()
        self.iteration += 1
        return info

    def update(self):
        raise NotImplementedError()

    def compute_blocksparse_parameters(self, number_of_cells):
        r"""Compute the centroids, the size of the cells and the keep
        matrix.

        Args:
            number_of_cells (int): The maximal number of cells. It will
                be rounded to the nearest lower power :math:`d`.

        Returns:

            (nb_cells, d) Tensor, (nb_cells,) Tensor,
            (nb_cells, nb_cells) BoolTensor:

            The centroids the size of the cells and the keep matrix.

        """
        if self.blocksparse:
            if self.d == 1 or self.d > 3:
                raise NotImplementedError()
            else:
                if isinstance(self.R, float):
                    nbox = (self.L / self.R).floor()
                else:
                    nbox = (self.L / self.R.min()).floor()
                if torch.prod(nbox) <= number_of_cells:
                    eps = self.L / nbox
                    centroids = uniform_grid_centroids(eps, self.L, self.d)
                    D = squared_distances_tensor(
                        centroids,
                        centroids,
                        self.L,
                        boundary_conditions=self.bc
                    )
                    keep = D < ((1.05 * eps) ** 2).sum()
                else:
                    eps0 = (self.L.prod() / number_of_cells) ** (1. / self.d)
                    while eps0 < self.R:
                        eps0 = 1.05 * eps0
                    nbox = (self.L / eps0).floor()
                    eps = self.L / nbox
                    centroids = uniform_grid_centroids(eps, self.L, self.d)
                    D = squared_distances_tensor(
                        centroids,
                        centroids,
                        self.L,
                        boundary_conditions=self.bc
                    )
                    keep = D < ((1.05 * eps) ** 2).sum()
        else:
            eps, keep, centroids = None, None, None
        return eps, keep, centroids

    def best_blocksparse_parameters(self, min_nb_cells,
                                    max_nb_cells, step=1, nb_calls=100,
                                    loop=1, set_best=True):
        r"""Returns the optimal block sparse parameters.

        This function measures the time to execute **nb_calls**
        calls to the :meth:`__next__` for different values of the block
        sparse parameters given by the number of cells to the power
        :math:`1/d`.

        Args:
            min_nb_cells (int): Minimal number of cells to the power
                :math:`1/d`.
            max_nb_cells (int): Maximal number of cells to the power
                :math:`1/d`.
            step (int, optional): Step between two numbers of cells.
                Default is 1.
            nb_calls (int, optional): Number of calls to the function
                :meth:`__next__` at each test. Default is 100.
            loop (int, optional): Number of loops. Default is 1.
            set_best (bool, optional): Set the block sparse parameters
                to the optimal ones. Default is True.

        Returns:
            tuple:

            * **fastest** `(int)`: The number of cells which gave the
                fastest result.
            * **nb_cells** `(list)`: The number of cells tested.
            * **average_simu_time** `(list)`: The average simulation
                times.
            * **simulation_time** `(numpy array)`: The total simulation
                times.

        """
        nb_cells = list(np.arange(min_nb_cells, max_nb_cells, step))
        simulation_time = np.zeros((loop, len(nb_cells)))
        for n, nb in enumerate(nb_cells):
            for i in range(loop):
                new_copy = copy.deepcopy(self)
                new_copy.blocksparse = True
                (new_copy.eps,
                 new_copy.keep,
                 new_copy.centroids) = self.compute_blocksparse_parameters(
                    nb ** new_copy.d)
                start = time.time()
                for k in range(nb_calls):
                    new_copy.__next__()
                end = time.time()
                simulation_time[i, n] = end-start
            progress = round(100*n/len(nb_cells), 2)
            sys.stdout.write('\r' + "Progress:" + str(progress) + "%")
        average_simu_time = np.mean(simulation_time, axis=0)
        fastest = nb_cells[np.argmin(average_simu_time)]
        if set_best:
            self.blocksparse=True
            (self.eps,
             self.keep,
             self.centroids) = self.compute_blocksparse_parameters(
                fastest ** self.d)
        return fastest, nb_cells, average_simu_time, simulation_time

    def add_target_method(self, name, method):
        r"""Add a method to the dictionary :attr:`target_method`.

        Args:
            name (str): The name of the method.
            method (func): A method.

        """
        self.target_method[name] = method

    def add_target_option_method(self, name, method):
        r"""Add an option to the dictionary :attr:`target_option_method`

        Args:
            name (str): The name of the option.
            method (func): An option.

        """
        self.target_option_method[name] = method

    def compute_target(self, which_target, which_options, **kwargs):
        r"""Compute a target and apply some options.

        Args:
            which_target (dict): Dictionary of the form::

                    {"name"       : "method_name",
                     "parameters" : {"param1" : param1,
                                     "param2" : param2,
                                      ...
                                    }
                    }

                The sub-dictionary ``"parameters"`` (eventually empty)
                contains the keyword arguments to be passed to the
                function found in the dictionary :attr:`target_method`
                with the keyword "method_name".
            which_options (dict): Dictionary of the form::

                    {"name_of_option1" : parameters_of_option1,
                     "name_of_option2" : parameters_of_option2,
                    ...
                    }

                The dictionary **`parameters_of_option1`** contains
                the keyword arguments to be passed to the method found
                in the dictionary :attr:`target_option_method` with the
                keyword "name_of_option1".
            **kwargs: Keywords arguments to be passed to the functions
                which compute the target and the options.

        Returns:
            Tensor:

            Compute a target using **which_target** and
            then apply the options in **which_options**
            successively.

        """
        target = self.target_method[which_target["name"]](
            **which_target["parameters"],
            **kwargs)
        if bool(which_options):
            for option_name in which_options:
                target = self.target_option_method[option_name](
                    target,
                    **which_options[option_name],
                    **kwargs)
        return target

    def linear_local_average(self, *to_average,
                             who=None, with_who=None,
                             isaverage=True,
                             kernel=lazy_interaction_kernel):
        r"""Fast computation of a linear local average.

        A linear local average is a matrix vector product between an
        interaction kernel (LazyTensor of size (M,N)) and a Tensor
        of size (N,dimU).

        Args:
            *to_average (Tensor): The quantities to average. The first
                dimension must be equal to :attr:`N`.
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            isaverage (bool, optional): Divide the result by :attr:`N`
                if **isaverage** is True. Default is True.
            kernel (func, optional): The function which returns the
                interaction kernel as a LazyTensor. Default is
                :func:`lazy_interaction_kernel`.

        Returns:
            tuple of Tensors:

            The linear local averages.

        """
        if who is None:
            who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        if with_who is None:
            with_who = torch.ones(self.N,
                                  dtype=torch.bool, device=self.pos.device)

        x = self.pos[who, :]
        y = self.pos[with_who, :]
        Rx = self.R[who] if type(x) == type(self.R) else self.R
        Ry = self.R[with_who] if type(y) == type(self.R) else self.R
        if self.axis is None:
            axis = None
        else:
            axis = self.axis[who, :]

        M, D = list(x.shape)
        N, D = list(y.shape)

        U = torch.cat(to_average, dim=1)
        dimensions = []
        for u in to_average:
            dimensions.append(u.shape[1])

        dimU = U.shape[1]
        U = U[with_who, :]

        if self.blocksparse:

            (x_sorted,
             y_sorted,
             nb_centro,
             labels_x,
             labels_y,
             permutation_x,
             ranges_ij,
             map_dummies) = block_sparse_reduction_parameters(
                x, y, self.centroids, self.eps, self.L, self.keep,
                where_dummies=False)

            if type(self.R) == type(x):
                Rx_plus = torch.cat((Rx, torch.zeros(nb_centro,
                                                     dtype=Rx.dtype,
                                                     device=Rx.device)))
                Rx_perm = Rx_plus[permutation_x]
                Ry_plus = torch.cat((Ry, torch.zeros(nb_centro,
                                                     dtype=Ry.dtype,
                                                     device=Ry.device)))
                Ry_perm, lryp = sort_clusters(Ry_plus, labels_y)
            else:
                Rx_perm = Rx
                Ry_perm = Ry

            U_plus = torch.cat((U, torch.zeros(nb_centro, dimU,
                                               dtype=U.dtype,
                                               device=U.device)))
            U_sorted, labels_u = sort_clusters(U_plus, labels_y)

            if self.angle < 2 * math.pi:
                dummy_axes = torch.randn((nb_centro, self.d),
                                         dtype = axis.dtype,
                                         device=axis.device)
                dummy_axes = dummy_axes / torch.norm(
                    dummy_axes, dim=1).reshape((nb_centro, 1))
                axis = torch.cat((axis, dummy_axes))
                axis = axis[permutation_x, :]

            K_ij = kernel(x=x_sorted, y=y_sorted,
                          Rx=Rx_perm, Ry=Ry_perm,
                          L=self.L, boundary_conditions=self.bc,
                          vision_angle=self.angle, axis=axis)

            K_ij.ranges = ranges_ij

            J_perm = K_ij @ U_sorted
            J_plus = J_perm[torch.argsort(permutation_x), :]
            # Remove the dummy particles
            J = J_plus[0:M, :]
            if isaverage:
                J = (1. / N) * J

        else:
            K_ij = kernel(x=x, y=y,
                          Rx=Rx, Ry=Ry,
                          L=self.L, boundary_conditions=self.bc,
                          vision_angle=self.angle, axis=axis)

            J = K_ij @ U
            if isaverage:
                J = (1. / self.N) * J
        return torch.split(J, dimensions, dim=1)

    def nonlinear_local_average(self, binary_formula,
                                arg1, arg2,
                                who=None, with_who=None,
                                isaverage=True,
                                kernel=lazy_interaction_kernel):
        r"""Fast computation of a nonlinear local average.

        A nonlinear local average is a reduction of the form

        .. math::
            \sum_j K_{ij} U_{ij}

        where :math:`K_{ij}` is an interaction kernel (
        LazyTensor of size (M,N)) and :math:`U_{ij}` is a
        LazyTensor of size (M,N,dim) computed using a binary
        formula.

        Args:
            binary_formula (func): Takes two arguments **arg1**
                and **arg2** and returns a LazyTensor of size
                (M,N,dim).
            arg1 ((M,D1) Tensor): First argument of the binary formula.
            arg2 ((N,D2) Tensor): Second argument of the binary
                formula.
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            isaverage (bool, optional): Divide the result by :attr:`N`
                if isaverage is True. Default is True.
            kernel (func, optional): The function which returns the
                interaction kernel as a LazyTensor. Default is
                :func:`lazy_interaction_kernel`.

        Returns:
            (M,dim) Tensor:

            The nonlinear local average.

        """
        if who is None:
            who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        if with_who is None:
            with_who = torch.ones(self.N,
                                  dtype=torch.bool, device=self.pos.device)

        x = self.pos[who, :]
        y = self.pos[with_who, :]
        Rx = self.R[who] if type(x) == type(self.R) else self.R
        Ry = self.R[with_who] if type(y) == type(self.R) else self.R
        if self.axis is None:
            axis = None
        else:
            axis = self.axis[who, :]

        M, D = list(x.shape)
        N, D = list(y.shape)

        if self.blocksparse:

            (x_sorted,
             y_sorted,
             nb_centro,
             labels_x,
             labels_y,
             permutation_x,
             ranges_ij,
             map_dummies) = block_sparse_reduction_parameters(
                x, y, self.centroids, self.eps, self.L, self.keep,
                where_dummies=True)

            if type(self.R) == type(x):
                Rx_plus = torch.cat((Rx, torch.zeros(nb_centro,
                                                     dtype=Rx.dtype,
                                                     device=Rx.device)))
                Rx_perm = Rx_plus[permutation_x]
                Ry_plus = torch.cat((Ry, torch.zeros(nb_centro,
                                                     dtype=Ry.dtype,
                                                     device=Ry.device)))
                Ry_perm, lryp = sort_clusters(Ry_plus, labels_y)
            else:
                Rx_perm = Rx
                Ry_perm = Ry

            if self.angle < 2 * math.pi:
                dummy_axes = torch.randn((nb_centro, self.d),
                                         dtype = axis.dtype,
                                         device=axis.device)
                dummy_axes = dummy_axes / torch.norm(
                    dummy_axes, dim=1).reshape((nb_centro, 1))
                axis = torch.cat((axis, dummy_axes))
                axis = axis[permutation_x, :]

            K_ij = kernel(x=x_sorted, y=y_sorted,
                          Rx=Rx_perm, Ry=Ry_perm,
                          L=self.L, boundary_conditions=self.bc,
                          vision_angle=self.angle, axis=axis)

            N1, D1 = list(arg1.size())
            N2, D2 = list(arg2.size())
            arg1_plus = torch.cat((arg1, torch.zeros(nb_centro, D1,
                                                     dtype=arg1.dtype,
                                                     device=arg1.device)))
            arg2_plus = torch.cat((arg2, torch.zeros(nb_centro, D2,
                                                     dtype=arg2.dtype,
                                                     device=arg2.device)))
            arg1_plus_sorted, la1 = sort_clusters(arg1_plus, labels_x)
            arg2_plus_sorted, la2 = sort_clusters(arg2_plus, labels_y)
            U_ij_plus = map_dummies * binary_formula(arg1_plus_sorted,
                                                     arg2_plus_sorted)

            KU = K_ij * U_ij_plus
            KU.ranges = ranges_ij
            J_perm = KU.sum(1)
            J_plus = J_perm[torch.argsort(permutation_x), :]
            # Remove the dummy particles
            J = J_plus[0:M, :]
            if isaverage:
                J = (1. / N) * J

        else:
            K_ij = kernel(x=x, y=y,
                          Rx=Rx, Ry=Ry,
                          L=self.L, boundary_conditions=self.bc,
                          vision_angle=self.angle, axis=axis)
            U_ij = binary_formula(arg1, arg2)
            KU = K_ij * U_ij
            J = KU.sum(1)
            if isaverage:
                J = (1. / N) * J

        return J

    def number_of_neighbours(self, who=None, with_who=None):
        r""" Compute the number of neighbours.

        The number of neighbours of ``x[i,:]`` is the number of ones in
        the row corresponding to ``x[i,:]`` in the interaction kernel.

        Args:
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.

        Returns:
            (M,) Tensor:

            The number of neighbours.

        """
        Nneigh, = self.linear_local_average(
            torch.ones((self.N, 1),
                       dtype=self.pos.dtype, device=self.pos.device),
            who=who, with_who=with_who, isaverage=False)
        return Nneigh.reshape(Nneigh.shape[0])

    def morse_target(self, Ca, la, Cr, lr, p=2, mass=1.,
                     who=None, with_who=None,
                     local=False,
                     isaverage=True,
                     kernel=lazy_interaction_kernel, **kwargs):
        r"""Compute the force exerted on each particle due to the Morse
        potential.

        Args:
            Ca (float): Attraction coefficient.
            la (float): Attraction length.
            Cr (float): Repulsion coefficient.
            lr (float): Repulsion length.
            p (int, optional): Exponent. Default is 2.
            mass ((N,) Tensor or float, optional): Mass of the
                particles. Default is 1.
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            local (bool, optional): If **local** is true then the sum
                only considers the y particles within the interaction
                kernel. Default is False.
            isaverage (bool, optional): If isaverage is True then divide
                the total force by N where N is the number of True in
                :Tensor"`with_who`. Default is True.
            kernel (func, optional): The interaction kernel to use.
                Default is :func:`lazy_interaction_kernel`
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor:

            The force exerted on each particle located at ``pos[who,:]``.

        """
        if who is None:
            who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        if with_who is None:
            with_who = torch.ones(self.N,
                                  dtype=torch.bool, device=self.pos.device)
        x = self.pos[who, :]
        y = self.pos[with_who, :]
        mx = mass[who] if type(x) == type(mass) else mass
        my = mass[with_who] if type(y) == type(mass) else mass
        if not local:
            lazy_morse_tensor = lazy_morse(x=x, y=y, Ca=Ca, la=la,
                                           Cr=Cr, lr=lr, p=p, mx=mx, my=my)
            if isaverage:
                return (1. / y.shape[0]) * lazy_morse_tensor.sum(axis=1)
            else:
                return lazy_morse_tensor.sum(axis=1)
        else:
            binary_formula = lambda x, y: lazy_morse(x=x, y=y, Ca=Ca, la=la,
                                                     Cr=Cr, lr=lr, p=p,
                                                     mx=mx, my=my)
            return self.nonlinear_local_average(
                binary_formula, x, y,
                who=who, with_who=with_who,
                isaverage=isaverage,
                kernel=kernel)

    def quadratic_potential_target(self, who=None, with_who=None,
                                   kernel=lazy_interaction_kernel,
                                   isaverage=True, **kwargs):
        r"""Compute the force exerted on each particle due to the
        quadratic potential.

        Args:
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            isaverage (bool, optional): If isaverage is True then divide
                the total force by N where N is the number of True in
                :Tensor"`with_who`. Default is True.
            kernel (func, optional): The interaction kernel to use.
                Default is :func:`lazy_interaction_kernel`
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor:

            The force exerted on each particle located at ``pos[who,:]``.

        """
        if who is None:
            who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        if with_who is None:
            with_who = torch.ones(self.N,
                                  dtype=torch.bool, device=self.pos.device)
        x = self.pos[who, :]
        y = self.pos[with_who, :]
        binary_formula = lambda x, y: lazy_quadratic(
            x=x, y=y, R=self.R, L=self.L, boundary_conditions=self.bc)
        return self.nonlinear_local_average(
            binary_formula, x, y,
            who=who, with_who=with_who,
            isaverage=isaverage,
            kernel=kernel)

    def overlapping_repulsion_target(self, who=None, with_who=None, **kwargs):
        r"""Compute the repulsion force exerted on each particle due to
        the overlapping.

        The force exerted on a particle located at :math:`x_i` derives
        from a logarithmic potential and is given by:

        .. math::

            F = \sum_j K(x_i-x_j) \left(\frac{(R_i+R_j)^2}{|x_i-x_j|^2} - 1\right) (x_i-x_j)

        where :math:`x_j` is the position of the other particles;
        :math:`R_i` and :math:`R_j` are the radii of the particles at
        :math:`x_i` and :math:`x_j` respectively and :math:`K` is the
        overlapping kernel.

        Note:
            in the present case, it is simpler to write explicitly
            the reduction formula but the same function could be
            implemented using a binary formula in
            :func:`nonlinear_local_average` with the kernel given by
            :func:`lazy_overlapping_kernel`.

        Args:
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor:

            The force exerted on each particle located at ``pos[who,:]``.

        """
        if who is None:
            who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        if with_who is None:
            with_who = torch.ones(self.N,
                                  dtype=torch.bool, device=self.pos.device)
        x = self.pos[who, :]  # (M,D)
        y = self.pos[with_who, :]  # (N,D)
        Rx = self.R[who] if type(self.R) == type(x) else self.R
        Ry = self.R[with_who] if type(self.R) == type(y) else self.R

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
        R_ij = R_ij * sq_dist.sign()
        sq_dist = sq_dist - (sq_dist.sign() - 1)
        F_ij = (((R_ij ** 2) / sq_dist - 1).relu()) * z_ij

        return F_ij.sum(axis=1)


"""Subclass of kinetic particles."""


class KineticParticles(Particles):
    r"""Subclass of kinetic particles.

    Attributes:
         vel ((N,d) Tensor): Velocities.

    """

    def __init__(self,
                 pos,
                 vel,
                 interaction_radius,
                 box_size,
                 vision_angle=2 * math.pi,
                 axis=None,
                 boundary_conditions='periodic',
                 block_sparse_reduction=False,
                 number_of_cells=1024):
        r"""

        Args:
            pos ((N,d) Tensor): Positions.
            vel ((N,d) Tensor): Velocities.
            interaction_radius (float or (N,) Tensor): Radius.
            box_size (float or list of size d): Box size.
            vision_angle (float, optional): Vision angle. Default is
                :math:`2\pi`.
            axis ((N,d) Tensor or None, optional): Axis of the
                particles. Default is None.
            boundary_conditions (list or str): Boundary conditions.
                Can be one of the following:

                - list of size :math:`d` containing 0 (periodic) or 1
                  (wall with reflecting boundary conditions) for each
                  dimension.

                - ``"open"`` : no boundary conditions.

                - ``"periodic"`` : periodic boundary conditions.

                - ``"spherical"`` : reflecting boundary conditions on the
                  sphere of radius ``L[0]/2`` and center **L**/2.
                Default is ``"periodic"``.
            block_sparse_reduction (bool, optional): Use block sparse
                reduction or not. Default is False.
            number_of_cells (int, optional): Maximum number of cells if
                **block_sparse_reduction** is True. Will be rounded to
                the nearest lower power :attr:`d`. Default is 1024.

        """
        self.vel = vel
        super().__init__(pos=pos,
                         interaction_radius=interaction_radius,
                         box_size=box_size,
                         vision_angle=vision_angle,
                         axis=axis,
                         boundary_conditions=boundary_conditions,
                         block_sparse_reduction=block_sparse_reduction,
                         number_of_cells=number_of_cells)
        self.add_target_method("normalised", self.normalised)
        self.add_target_method("motsch_tadmor", self.motsch_tadmor)
        self.add_target_method("mean_field", self.mean_field)
        self.add_target_method("max_kappa", self.max_kappa)
        self.add_target_method("nematic", self.nematic)
        self.add_target_option_method("bounded_angular_velocity",
                                      self.bounded_angular_velocity)
        self.add_target_option_method("pseudo_nematic", self.pseudo_nematic)

    @property
    def axis(self):
        r"""
        If a None axis is given, then the default axis is the
        normalised velocity.

        Returns:
            (N,d) Tensor

        """
        if self.axis_is_none:
            return self.vel / torch.norm(self.vel, dim=1).reshape((self.N, 1))
        else:
            return self.__axis

    @axis.setter
    def axis(self, value):
        if self.axis_is_none:
            """DO NOTHING"""
        else:
            self.__axis = value

    @property
    def order_parameter(self):
        r"""Compute the norm of the average velocity.

        Returns:
            float:

            The order parameter.

        """
        vec_phi = (1. / self.N) * self.vel.sum(0)
        phi = torch.norm(vec_phi)
        return float(phi)

    def check_boundary(self):
        r"""Update the positions and velocities of the particles which
           are outside the boundary.

        """
        if self.bc == 'periodic':
            for i in range(self.d):
                self.pos[:, i] = torch.remainder(self.pos[:, i], self.L[i])
        elif type(self.bc) == list:
            for i in range(self.d):
                if self.bc[i] == 1:
                    inf0 = self.pos[:, i] < 0
                    supL = self.pos[:, i] > self.L[i]
                    self.pos[inf0, i] = -self.pos[inf0, i]
                    self.pos[supL, i] = 2 * self.L[i] - self.pos[supL, i]

                    self.vel[inf0, i] = -self.vel[inf0, i]
                    self.vel[supL, i] = -self.vel[supL, i]
                elif self.bc[i] == 0:
                    self.pos[:, i] = torch.remainder(self.pos[:, i], self.L[i])
        elif self.bc == 'spherical':
            # Translate the center to the origin and compute the
            # radius = L/2.
            x = self.pos - self.L / 2
            r = self.L[0] / 2
            out = (x ** 2).sum(1) > (r ** 2)
            N_out = out.sum()
            if N_out > 0:
                x_out = x[out, :]
                v_out = self.vel[out, :]
                # Find x0 and c0 such that x = x0 + c0*vel with |x0|=r
                xdotv = (x_out * v_out).sum(1)
                sqnorm_v_out = (v_out ** 2).sum(1)
                sqnorm_x_out = (x_out ** 2).sum(1)
                D0 = (xdotv ** 2
                      - sqnorm_v_out * (sqnorm_x_out - r ** 2)
                      ).abs()  # Take the absolute value to ensure >=0
                c0 = 1. / sqnorm_v_out * (xdotv - D0.sqrt())
                c0 = c0.reshape((N_out, 1))
                x0 = x_out - c0 * v_out
                # Find x1 and c1 such that x = x1+c1*x0 with |x1|=r
                xdotx0 = (x_out * x0).sum(1)
                D1 = (xdotx0 ** 2 - (r ** 2) * (sqnorm_x_out - r ** 2)).abs()
                c1 = (r ** (-2)) * (xdotx0 - D1.sqrt())
                c1 = c1.reshape((N_out, 1))
                # Find c2 such that v = v2+c2*x0 and |v|=|v2|
                c2 = 2. * (r ** (-2)) * ((v_out * x0).sum(1))
                c2 = c2.reshape((N_out, 1))
                # Construct the reflections
                x_reflect = x_out - 2 * c1 * x0
                v_reflect = v_out - c2 * x0
                v_reflect = v_reflect / torch.norm(v_reflect,
                                                   dim=1).reshape((N_out, 1)) \
                            * sqnorm_v_out.sqrt().reshape((N_out, 1))
                # Update the positions and velocity
                self.pos[out, :] = x_reflect + self.L / 2
                self.vel[out, :] = v_reflect
                # Stability and emergency procedure....
                self.pos = torch.max(self.pos,
                                     torch.tensor([0.],
                                                  dtype=self.pos.dtype,
                                                  device=self.pos.device))
                self.pos = torch.min(self.pos, self.L[0])
                x = self.pos - self.L / 2
                r = self.L[0] / 2
                out = (x ** 2).sum(1) > (r ** 2)
                N_out = out.sum()
                if N_out > 0:
                    if N_out > 42:
                        print('Warning!'+str(N_out)+'particles out')
                    self.pos[out, :] = self.L/2 + torch.zeros(
                        (N_out, self.d),
                        dtype=self.pos.dtype,
                        device=self.pos.device
                    )

    def normalised(self, kappa, who=None, with_who=None, **kwargs):
        r"""The normalised average velocity.

        Given a system of particles with positions :math:`(x_i)_i` and
        velocities :math:`(v_i)_i`, the normalised average velocity
        around particle :math:`(x_i,v_i)` is

        .. math::

            \Omega_i = \frac{\sum_j K(x_i-x_j) v_j}{\left|\sum_j K(x_i-x_j) v_j\right|},

        where :math:`K` is the interaction kernel.

        Args:
            kappa ((1,) Tensor or (M,) Tensor): The concentration
                parameter. If the size of kappa is bigger than 1 then
                its size must be equal to the number of True values in
                **who**.
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor:

            The normalised averaged velocity multiplied by the
            concentration parameter.

        """
        J, = self.linear_local_average(self.vel, who=who, with_who=with_who)
        Omega = J / torch.norm(J, dim=1).reshape((J.shape[0], 1))
        return kappa * Omega

    def motsch_tadmor(self, kappa, who=None, with_who=None, **kwargs):
        r"""The average velocity of the neighbours only.

        Given a system of particles with positions :math:`(x_i)_i` and
        velocities :math:`(v_i)_i`, the average velocity
        around particle :math:`(x_i,v_i)` with the Motsch-Tadmor
        normalisation is:

        .. math::

            \Omega_i = \frac{\sum_j K(x_i-x_j) v_j}{\sum_j K(x_i-x_j)},

        where :math:`K` is the interaction kernel.

        Args:
            kappa ((1,) Tensor or (M,) Tensor): The concentration
                parameter. If the size of kappa is bigger than 1 then
                its size must be equal to the number of True values in
                **who**.
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor:

            The average velocity of the neighbours multiplied
            by the concentration parameter.

        """
        justN = torch.ones((self.N, 1),
                           dtype=self.pos.dtype, device=self.pos.device)
        J, N_neighbours = self.linear_local_average(self.vel, justN,
                                                    who=who, with_who=with_who)
        return kappa * J / N_neighbours

    def mean_field(self, kappa, who=None, with_who=None, **kwargs):
        r"""Non-normalised average velocity.

        Given a system of particles with positions :math:`(x_i)_i` and
        velocities :math:`(v_i)_i`, the mean-field average velocity
        around particle :math:`(x_i,v_i)` (without normalisation) is:

        .. math::

            J_i = \frac{1}{N}\sum_j K(x_i-x_j) v_j,

        where :math:`K` is the interaction kernel.

        Args:
            kappa ((1,) Tensor or (M,) Tensor): The concentration
                parameter. If the size of kappa is bigger than 1 then
                its size must be equal to the number of True values in
                **who**.
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor:

            The mean-field average velocity multiplied by a scaling
            parameter and by the concentration parameter. The scaling
            parameter is equal to the volume of the box divided by
            the volume of the ball of radius :attr:`R` in dimension
            :attr:`d`.

        """
        scaling = (self.L / self.R) ** self.d * 1. / volume_ball(self.d)
        J, = self.linear_local_average(self.vel, who=who, with_who=with_who)
        J = scaling * J
        return kappa * J

    def max_kappa(self, kappa, kappa_max, who=None, with_who=None, **kwargs):
        r""" The mean-field target with a norm threshold.

        Given a system of particles with positions :math:`(x_i)_i` and
        velocities :math:`(v_i)_i`, the mean-field average velocity
        around particle :math:`(x_i,v_i)` with a cutoff is:

        .. math::

            \Omega_i = \frac{1}{\frac{1}{\kappa}+\frac{|J_i|}{\kappa_\mathrm{max}}} J_i

        where

        .. math::

            J_i = \frac{1}{N}\sum_j K(x_i-x_j) v_j,

        and :math:`K` is the interaction kernel multiplied by the
        scaling parameter. The scaling parameter is equal to the
        volume of the box divided by the volume of the ball of radius
        :attr:`R` in dimension :attr:`d`. The norm of :math:`\Omega_i`
        is thus bounded by :math:`\kappa_\mathrm{max}`.

        Args:
            kappa ((1,) Tensor or (M,) Tensor): The concentration
                parameter. If the size of kappa is bigger than 1 then
                its size must be equal to the number of True values in
                **who**.
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor:

            The cutoff average velocity.

        """
        scaling = (self.L / self.R) ** self.d * 1. / volume_ball(self.d)
        J, = self.linear_local_average(self.vel, who=who, with_who=with_who)
        J = scaling * J
        normJ = torch.norm(J, dim=1).reshape((J.shape[0], 1))
        return 1. / ((1. / kappa) + (normJ / kappa_max)) * J

    def nematic(self, kappa, who=None, with_who=None, **kwargs):
        r"""Normalised average of the nematic velocities.

        Given a system of particles with positions :math:`(x_i)_i` and
        velocities :math:`(v_i)_i`, the nematic average velocity
        around particle :math:`(x_i,v_i)` is:

        .. math::

            \Omega_i = (v_i\cdot \overline{\Omega}_i)\overline{\Omega}_i,

        where :math:`\overline{\Omega}_i` is any unit eigenvector
        associated to the maximal eigenvalue of the average Q-tensor:

        .. math::

            Q_i = \sum_j K(x_i-x_j) \left(v_j\otimes v_j - \frac{1}{d} I_d\right),

        where :math:`K` is the interaction kernel.

        Args:
            kappa ((1,) Tensor or (M,) Tensor): The concentration
                parameter. If the size of kappa is bigger than 1 then
                its size must be equal to the number of True values in
                **who**.
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor:

            The nematic average velocity multiplied by the concentration
            parameter.

        """
        if who is None:
            who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        Q = torch.einsum('ijl,ilk->ijk',
                         self.vel[:, :, None], self.vel[:, None, :]) \
            - (1. / self.d) * torch.eye(self.d,
                                        dtype=self.vel.dtype,
                                        device=self.vel.device)
        reshapeQ = Q.reshape(self.N, self.d ** 2)
        J, = self.linear_local_average(reshapeQ, who=who, with_who=with_who)
        M = J.shape[0]
        J = J.reshape(M, self.d, self.d)
        if self.d == 2:
            targets = maximal_eigenvector_dim2(J)
        else:
            eivel, eivec = torch.linalg.eigh(J.to('cpu'))
            targets = eivec[:, :, -1]
            targets = targets.to(J.device)
        dot = (self.vel[who, :] * targets).sum(1).reshape((M, 1))
        targets = dot * targets
        return kappa * targets

    def bounded_angular_velocity(self, target, angvel_max, dt,
                                 who=None, **kwargs):
        r"""Bounded angle between the velocity and the target.

        The maximum angle between the velocity and the new target cannot
        exceed the value ``angvel_max * dt``. The output tensor
        **new_target** is of the form::

            new_target = a * vel + b * target

        where :math:`a,b>0` are such that the cosine between
        **new_target** and :attr:`vel` is equal to::

            max_cos := max(cos(angvel_max * dt), cos_target)

        where **cos_target** is the cosine between **target** and
        :attr:`vel`. The solution is::

            b = (1 - max_cos ** 2) / (1 - cos_target ** 2)
            a = |target| / |vel| * (max_cos - b * cos_target)

        Args:
            target ((M,d) Tensor): Target.
            angvel_max (float): Maximum angular velocity.
            dt (float): Interval of time.
            who ((N,) BoolTensor, optional): The velocities of the
                particles associated to the target are ``vel[who,:]``
                (or :attr:`vel` if **who** is None). Default is None.
            **kwargs: Arbitrary keywords arguments.

        Returns:
            (M,d) Tensor:

            The new target.

        """
        if who is None:
            who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        M = target.shape[0]
        vel = self.vel[who, :]
        norm_vel = torch.norm(vel, dim=1).reshape((M, 1))
        norm_target = torch.norm(target, dim=1).reshape((M, 1))

        cos_max = math.cos(angvel_max * dt)
        dot = (vel * target).sum(axis=1).reshape((M, 1))
        cos = dot / (norm_vel * norm_target)
        max_cos = dot / (norm_vel * norm_target)
        max_cos[max_cos <= cos_max] = cos_max

        # Add .01 for stability
        b = torch.sqrt((1.01 - max_cos ** 2) / (1.01 - cos ** 2))
        a = (norm_target / norm_vel) * (max_cos - b * cos)

        new_target = a * vel + b * target

        return new_target

    def pseudo_nematic(self, target, who=None, **kwargs):
        r"""Choose the closest to the velocity between the
        target and its opposite.

        Args:
            target ((M,d) Tensor): Target.
            who ((N,) BoolTensor, optional): The velocities of the
                particles associated to the target are ``vel[who,:]``
                (or :attr:`vel` if **who** is None). Default is None.
            **kwargs: Arbitrary keywords arguments.

        Returns:
            (M,d) Tensor:

            Take the opposite of **target** if the angle with the
            velocity is larger than :math:`\pi/2`.

        """
        if who is None:
            who = torch.ones(self.N, dtype=torch.bool, device=self.pos.device)
        dot = (self.vel[who, :] * target).sum(axis=1)
        target[dot < 0] = -target[dot < 0]
        return target


"""The subclass of body-oriented particles."""


class BOParticles(Particles):
    r"""3D particle with a body-orientation in :math:`SO(3)`.

    A body-orientation is a rotation matrix is :math:`SO(3)` or
    equivalently a unit quaternion.
    If :math:`q = (x, y, z, t)` is a unit quaternion, the associated
    rotation matrix is :math:`A = [e_1, e_2, e_3]` where the column
    vectors are defined by:

    .. math::

        e_1 = (x^2 + y^2 - z^2 + t^2, 2(xt+yz), 2(yt-xz))^T

    .. math::
        e_2 = (2(yz-xt), x^2 - y^2 + z^2 - t^2, 2(xy+zt))^T

    .. math::
        e_3 = (2(xz+yt), 2(zt-xy), x^2 - y^2 - z^2 + t^2)^T

    Conversely, if :math:`A` is the rotation of angle :math:`\theta`
    around the axis :math:`u = (u_1, u_2, u_3)^T` then the associated
    unit quaternion is:

    .. math::

        q = \cos(\theta/2) + \sin(\theta/2)(iu_1 + ju_2 + ku_3).

    or its opposite.

    Attributes:
        bo ((N,4) Tensor): Body-orientation stored as a unit quaternion.

    """

    def __init__(self, pos, bo,
                 interaction_radius, box_size,
                 vision_angle=2 * math.pi, axis=None,
                 boundary_conditions='periodic',
                 block_sparse_reduction=False, number_of_cells=1024):
        r"""

        Args:
            pos ((N,3) Tensor): Positions.
            bo ((N,4) Tensor): Body-orientations (unit quaternions).
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

                - ``"spherical"`` : reflecting boundary conditions on the
                  sphere of radius ``L[0]/2`` and center **L**/2.

                Default is ``"periodic"``.
            block_sparse_reduction (bool, optional): Use block sparse
                reduction or not. Default is False
            number_of_cells (int, optional): Maximum number of cells if
                **block_sparse_reduction** is True. Will be rounded to
                the nearest lower power :attr:`d`. Default is 1024.

        """
        self.bo = bo
        super().__init__(pos,
                         interaction_radius,
                         box_size,
                         vision_angle=vision_angle, axis=axis,
                         boundary_conditions=boundary_conditions,
                         block_sparse_reduction=block_sparse_reduction,
                         number_of_cells=number_of_cells)
        self.add_target_method("normalised", self.normalised)

    @property
    def vel(self):
        r"""The velocity is the first column vector of the
        body-orientation matrix.

        Returns:
            (N,3) Tensor:

            The velocity.

        """
        x = self.bo[:, 0]
        y = self.bo[:, 1]
        z = self.bo[:, 2]
        t = self.bo[:, 3]
        e1 = torch.stack(
            (x ** 2 + y ** 2 - z ** 2 - t ** 2,
             2 * (x * t + y * z),
             2 * (y * t - x * z)), dim=1)
        return e1

    @property
    def axis(self):
        r""" If a None axis is given, then the default axis is the
        (normalised) velocity.

        Returns:
            (N,3) Tensor:

        """
        if self.axis_is_none:
            return self.vel
        else:
            return self.__axis

    @axis.setter
    def axis(self, value):
        if self.axis_is_none:
            """DO NOTHING"""
        else:
            self.__axis = value

    @property
    def orth_basis(self):
        r"""An othogonal basis of the orthogonal plane of the velocity.

        Returns:
            tuple:

            * **e2** *((N,3) Tensor)*: Second column vector of the
              body-orientation matrix.

            * **e3** *((N,3) Tensor)*: Third column vector of the
              body-orientation matrix.

        """
        x = self.bo[:, 0]
        y = self.bo[:, 1]
        z = self.bo[:, 2]
        t = self.bo[:, 3]
        e2 = torch.stack(
            (2 * (y * z - x * t),
             x ** 2 - y ** 2 + z ** 2 - t ** 2,
             2 * (x * y + z * t)), dim=1)
        e3 = torch.stack(
            (2 * (x * z + y * t),
             2 * (z * t - x * y),
             x ** 2 - y ** 2 - z ** 2 + t ** 2), dim=1)
        return e2, e3

    @property
    def qtensor(self):
        r"""Compute the normalised uniaxial Q-tensors of the particles.

        If :math:`q` is a unit quaternion seen as a colum vector of
        dimension 4, the normalised uniaxial Q-tensor is defined by:

        .. math::

            Q = q \otimes q - \frac{1}{4}I_4

        where :math:`\otimes` is the tensor product and :math:`I_4` the
        identity matrix of size 4.

        Returns:
            (N,4,4) Tensor:

            Q-tensors of the N particles.

        """
        Q = torch.einsum('ijl,ilk->ijk',
                         self.bo[:, :, None], self.bo[:, None, :]) \
            - .25 * torch.eye(4, dtype=self.bo.dtype, device=self.bo.device)
        return Q

    @property
    def omega_basis(self):
        r"""Local frame associated to the direction of motion.

        The unit vector :math:`\Omega` is the direction of motion
        (it is equal to :attr:`vel`) and
        :math:`[\Omega, e_\theta, e_\varphi]` is the local orthomormal
        frame associated to the spherical cooordinates.

        Returns:
            tuple:

            * **p** *((N,3) Tensor)*: :math:`e_\varphi`.
            * **q** *((N,3) Tensor)*: :math:`-e_\theta`.
        """
        Omega = self.vel
        phi = torch.atan2(Omega[:, 1], Omega[:, 0])
        p = torch.cat(
            (-torch.sin(phi).reshape((self.N, 1)),
             torch.cos(phi).reshape((self.N, 1)),
             torch.zeros((self.N, 1), dtype=phi.dtype, device=phi.device)),
            dim=1)
        q = torch.cross(Omega, p)
        return p, q

    @property
    def u_in_omega_basis(self):
        r"""Coordinates of :math:`u` in the frame :math:`[\Omega, p, q]`
        given by :attr:`omega_basis` where :math:`u` is the second
        column vector of the body-orientation matrix."""
        p, q = self.omega_basis
        u, v = self.orth_basis
        up = (u * p).sum(dim=1)
        uq = (u * q).sum(dim=1)
        return up, uq

    def normalised(self, who=None, with_who=None, **kwargs):
        r"""Projection of the average body-orientation on :math:`SO(3)`.

        Given a system of particles with positions :math:`(x_i)_i` and
        body-orientation matrices :math:`(A_i)_i`, the mean-field average
        body-orientation around the particle :math:`(x_i,A_i)` is
        is the arithmetic average:

        .. math::

            J_i = \sum_j K(x_i - x_j) A_j

        where :math:`K` is the interaction kernel. The projection on
        :math:`SO(3)` is

        .. math::

            P_{SO(3)}(J_i) = argmax_{A\in SO(3)}( A \cdot J_i )

        for the dot product

        .. math::

            A \cdot B = \frac{1}{2} Tr(A^T B).

        The unit quaternion associated to :math:`P_{SO(3)}(J_i)` is any
        eigenvector associated to the maximal eigenvalue of the average
        Q-tensor:

        .. math::

            \overline{Q}_i = \sum_j K(x_i - x_j) Q_j

        where :math:`Q_j` is the Q-tensor of particle :math:`(x_j,A_j)`
        given by :attr:`qtensor`.

        Args:
            who ((N,) BoolTensor or None, optional): The rows of
                the interaction kernel is ``pos[who,:]`` (or
                :attr:`pos` if **who** is None). Default is None.
            with_who ((N,) BoolTensor or None, optional): The
                columns of the interaction kernel is
                ``pos[with_who,:]`` (or :attr:`pos` if **with_who** is
                None). Default is None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            (M,4) Tensor:

            The unit quaternion associated to the projection of the
            mean-field average body-orientation on :math:`SO(3)`.

        """
        reshapeQ = self.qtensor.reshape(self.N, 16)
        J, = self.linear_local_average(reshapeQ, who=who, with_who=with_who)
        M = J.shape[0]
        J = J.reshape(M, 4, 4)
        # Faster on the cpu !
        eivel, eivec = torch.linalg.eigh(J.to('cpu'))
        targets = eivec[:, :, 3]
        # Back to the gpu.
        targets = targets.to(J.device)
        return targets

    ### Some statistical quantities ###

    @property
    def theta(self):
        r"""Polar angle of the direction of motion"""
        Omega = self.vel.sum(0)
        Omega = Omega / torch.norm(Omega)
        return float(torch.acos(Omega[2]))

    @property
    def phi(self):
        r"""Azimuthal angle of the direction of motion"""
        Omega = self.vel.sum(0)
        Omega = Omega / torch.norm(Omega)
        phi_pi = float(torch.atan2(Omega[1], Omega[0]))
        return np.mod(phi_pi, 2 * np.pi)

    def local_vel(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1,
                  who=None):
        if who is None:
            index = (self.pos[:, 0] >= xmin and self.pos[:, 0] <= xmax) \
                    and (self.pos[:, 0] >= xmin and self.pos[:, 0] <= xmax) \
                    and (self.pos[:, 0] >= xmin and self.pos[:, 0] <= xmax)
        else:
            index = who
        Omega = self.vel[index, :].sum(0)
        order_parameter = torch.norm(Omega) / len(self.vel[index, 0])
        Omega = Omega / torch.norm(Omega)
        theta = float(torch.acos(Omega[2]))
        phi = np.mod(float(torch.atan2(Omega[1], Omega[0])), 2 * np.pi)
        return order_parameter, theta, phi

    def twist(self, K, axis, return_slice_nb=None, local_vel=False):
        up, uq = self.u_in_omega_basis
        if axis == 'x':
            if self.L.shape == torch.Size([]):
                eps = self.L / K
            else:
                eps = self.L[0] / K
            index = (self.pos[:, 0] / eps).floor().int()
        elif axis == 'y':
            if self.L.shape == torch.Size([]):
                eps = self.L / K
            else:
                eps = self.L[1] / K
            index = (self.pos[:, 1] / eps).floor().int()
        elif axis == 'z':
            if self.L.shape == torch.Size([]):
                eps = self.L / K
            else:
                eps = self.L[2] / K
            index = (self.pos[:, 2] / eps).floor().int()
        else:
            raise ValueError('axis must be x, y or z')
        index = torch.max(index, torch.tensor(0, device=index.device))
        index = torch.min(index, torch.tensor(K - 1, device=index.device))
        slice_p = torch.zeros(K, dtype=up.dtype, device=up.device)
        slice_q = torch.zeros(K, dtype=uq.dtype, device=uq.device)
        if local_vel:
            op = torch.zeros(K, dtype=up.dtype, device=up.device)
            theta_loc = torch.zeros(K, dtype=up.dtype, device=up.device)
            phi_loc = torch.zeros(K, dtype=up.dtype, device=up.device)
        for k in range(K):
            Nk = torch.sum(index == k)
            slice_p[k] = torch.sum(up[index == k]) / Nk
            slice_q[k] = torch.sum(uq[index == k]) / Nk
            if local_vel:
                who_k = (index == k)
                opk, thk, phk = self.local_vel(who=who_k)
                op[k] = opk
                theta_loc[k] = thk
                phi_loc[k] = phk
        if return_slice_nb is None:
            if local_vel:
                return slice_p, slice_q, op, theta_loc, phi_loc
            else:
                return slice_p, slice_q
        else:
            if local_vel:
                return (slice_p, slice_q,
                        up[index == return_slice_nb],
                        uq[index == return_slice_nb],
                        op, theta_loc, phi_loc)
            else:
                return (slice_p, slice_q,
                        up[index == return_slice_nb],
                        uq[index == return_slice_nb])

    @property
    def local_order(self):
        def alpha(arg1, arg2):
            arg_i = LazyTensor(arg1[:, None, :])
            arg_j = LazyTensor(arg2[None, :, :])
            return ((arg_i * arg_j).sum(-1)) ** 2

        Phi = self.nonlinear_local_average(alpha, self.bo, self.bo)
        Nneigh = self.number_of_neighbours()
        Phi = Phi - (1. / self.N)
        Nneigh = torch.max(Nneigh - (1. / self.N),
                           torch.tensor(1. / self.N,
                                        dtype=Nneigh.dtype,
                                        device=Nneigh.device))
        order_i = Phi * (Nneigh ** (-1))
        return (1 / self.N) * order_i.sum().item()

    @property
    def global_order(self):
        r"""Returns

         .. math::

            \frac{1}{ N(N-1)} \sum_{i,j} (q_i \cdot q_j)^2.

        """
        bo_i = LazyTensor(self.bo[:, None, :])
        bo_j = LazyTensor(self.bo[None, :, :])
        bo_ij = ((bo_i * bo_j).sum(-1)) ** 2
        order_i = 1. / (self.N - 1.) * (bo_ij.sum(1) - 1.)
        return (1. / self.N) * order_i.sum().item()

    @property
    def order_parameter(self):
        return self.global_order

    def build_reflection_quat(self, who, axis):
        r"""

        Returns the quaternion associated to the rotation of angle
        :math:`2\alpha` and axis :math:`n` where :math:`\alpha` is the
        angle between ``vel[who,:]`` and the plane  orthogonal to
        **axis** and :math:`n` is the element of this plane obtained by
        a rotation of :math:`\pi/2` around **axis** of the normalised
        projection of ``vel[who,:]``.

        Args:
            who ((N,) BoolTensor)
            axis (int): The axis number either 0 (:math:`x` axis), 1
                (:math:`y` axis) or 2 (:math:`z` axis).

        Returns:
            (M,4) Tensor:

            Rotation (unit quaternion).

        """
        Omega = self.vel[who, :]
        Om1 = Omega[:, 0]
        Om2 = Omega[:, 1]
        Om3 = Omega[:, 2]
        K = Omega.shape[0]
        if axis == 0:
            q0 = torch.sqrt(Om2 ** 2 + Om3 ** 2)
            q = torch.stack(
                (q0,
                 torch.zeros(K, dtype=self.bo.dtype, device=self.bo.device),
                 -Om1 * Om3 / q0,
                 Om1 * Om2 / q0), dim=1)
        elif axis == 1:
            q0 = torch.sqrt(Om3 ** 2 + Om1 ** 2)
            q = torch.stack(
                (q0,
                 -Om2 * Om3 / q0,
                 torch.zeros(K, dtype=self.bo.dtype, device=self.bo.device),
                 Om2 * Om1 / q0), dim=1)
        elif axis == 2:
            q0 = torch.sqrt(Om1 ** 2 + Om2 ** 2)
            q = torch.stack(
                (q0,
                 -Om3 * Om2 / q0,
                 Om3 * Om1 / q0,
                 torch.zeros(K, dtype=self.bo.dtype, device=self.bo.device)),
                dim=1)
        else:
            raise ValueError('axis must be 0, 1 or 2.')
        return q

    def check_boundary(self):
        r"""Update the positions and body-orientations of the particles
           which are outside the boundary.

        """
        if self.bc == 'periodic':
            for i in range(self.d):
                self.pos[:, i] = torch.remainder(self.pos[:, i], self.L[i])
        elif type(self.bc) == list:
            for i in range(self.d):
                if self.bc[i] == 1:
                    inf0 = self.pos[:, i] < 0
                    supL = self.pos[:, i] > self.L[i]
                    self.pos[inf0, i] = torch.max(
                        -self.pos[inf0, i],
                        torch.tensor([0.],
                                     dtype=self.pos.dtype,
                                     device=self.pos.device))
                    self.pos[supL, i] = torch.min(
                        2 * self.L[i] - self.pos[supL, i],
                        self.L[i])

                    q_infzero = self.build_reflection_quat(who=inf0, axis=i)
                    self.bo[inf0, :] = quat_mult(q_infzero,
                                                 self.bo[inf0, :])
                    q_supL = self.build_reflection_quat(who=supL, axis=i)
                    self.bo[supL, :] = quat_mult(q_supL,
                                                 self.bo[supL, :])
                elif self.bc[0] == 0:
                    self.pos[:, i] = torch.remainder(self.pos[:, i], self.L[i])
                else:
                    raise NotImplementedError()
            # Normalise for numerical stability
            self.bo = self.bo / torch.sqrt(
                torch.sum(self.bo ** 2, dim=1)).reshape((self.N, 1))
        else:
            raise NotImplementedError()
