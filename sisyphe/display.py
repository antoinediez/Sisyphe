import math
import torch
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection
import sys
import os
import pickle


def save(simu, frames, attributes, attributes_alltimes, Nsaved=None,
         save_file=True, file_name="data"):
    """Basic saving function.

    Args:
        simu (Particles): A model.
        frames (list): List of times (float) to save.
        attributes (list): List of strings containing the names of the
            attributes to save at the times in the list **frames**.
        attributes_alltimes (list): List of strings containing the
            names of the attributes which will be saved at each
            iteration until the last time in frame.
        Nsaved (int, optional): Default is None which means that all
            the particles will be saved.
        save_file: (bool, optional): Default is True which means that
            a pickle file will be generated and saved in the current
            directory.
        file_name (str, optional): Default is ``"data"``.


    Returns:
        dict:

        A dictionary which contains the entry keywords ``"time"``,
        ``"frames"`` and all the elements of the lists **attributes**
        and **attributes_alltimes**. The dictionary is saved in the
        current directory.

    """
    frames = sorted(frames)
    if Nsaved is None:
        N = simu.N
    else:
        N = Nsaved
    data = {"time" : [0],
            "frames" : [0]}
    for att in attributes_alltimes:
        if type(att) is str:
            x = getattr(simu, att)
            att_name = att
        else:
            x = att(simu)
            att_name = att.__name__
        if torch.is_tensor(x):
            if x.shape[0] == simu.N:
                x = x[:N].clone()
        data[att_name] = [x]
    for att in attributes:
        if type(att) is str:
            x = getattr(simu, att)
            att_name = att
        else:
            x = att(simu)
            att_name = att.__name__
        if torch.is_tensor(x):
            if x.shape[0] == simu.N:
                x = x[:N].clone()
        data[att_name] = [x]
    t = 0
    tmax = len(frames) - 1
    percent = 0
    while t < tmax+1:
        simu.__next__()
        data["time"].append(data["time"][-1] + simu.dt)
        if data["time"][-1] / frames[tmax] >= percent / 100.:
            sys.stdout.write('\r' + "Progress:" + str(percent) + "%")
            sys.stdout.flush()
            percent += 1
        if Nsaved is None:
            N = simu.N
        else:
            N = Nsaved
        for att in attributes_alltimes:
            if type(att) is str:
                x = getattr(simu, att)
                att_name = att
            else:
                x = att(simu)
                att_name = att.__name__
            if torch.is_tensor(x):
                if x.shape[0] == simu.N:
                    x = x[:N].clone()
            data[att_name].append(x)
        if abs(data["time"][-1] >= frames[t]):
            data["frames"].append(data["time"][-1])
            for att in attributes:
                if type(att) is str:
                    x = getattr(simu, att)
                    att_name = att
                else:
                    x = att(simu)
                    att_name = att.__name__
                if torch.is_tensor(x):
                    if x.shape[0] == simu.N:
                        x = x[:N].clone()
                data[att_name].append(x)
            t += 1
    if save_file:
        pickle.dump(data, open(file_name + ".p", "wb"))
    return data


def display(position, velocity):
    # Quiver plot in dimension 2
    cpu_pos = position.cpu()
    cpu_vel = velocity.cpu()
    U = cpu_vel[:, 0]
    V = cpu_vel[:, 1]
    plot = plt.quiver(cpu_pos[:, 0], cpu_pos[:, 1], U, V)
    return plot


def display_kinetic_particles(simu, time, N_dispmax=None,
                              order=False, color=False,
                              show=True, figsize=(6,6),
                              save=False, path='simu'):
    """Basic quiver plot in dimension 2 and scatter plot in dimension 3.

    Args:
        simu (Particles): A model.
        time (list): List of times to plot.
        N_dispmax (int, optional): Default is None.
        order (bool, optional): Compute the order parameter or not.
            Default is False.
        color (bool, optional): The color of the particle depends on
            the velocity angle (dimension 2). Default is False.
        show (bool, optional): Show the plot. Default is True.
        figsize (tuple, optional): Figure size, default is (6,6).
        save (bool, optional): Save the plots. Default is False.
        path (str, optional): The plots will be saved in the directory
            ``'./path/frames'``.

    Returns:
        list, list:

        Times and order parameter

    """
    t = 0
    tmax = len(time) - 1
    percent = 0
    op = []
    real_time = []
    if save:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, path, r'frames')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
    for it, info in enumerate(simu):
        if N_dispmax is None:
            N_displayed = simu.N
        else:
            N_displayed = min(N_dispmax, simu.N)
        real_time.append(it * simu.dt)
        if order:
            op.append(simu.order_parameter)
        if it == np.floor((percent / 100) * time[tmax] / simu.dt):
            sys.stdout.write('\r' + "Progress:" + str(percent) + "%")
            sys.stdout.flush()
            percent += 1
        if abs(it * simu.dt - time[t]) < simu.dt:
            if simu.d == 2:
                f = plt.figure(t, figsize=figsize)
                ax = f.add_subplot(111)
                x = simu.pos[:N_displayed, 0]
                y = simu.pos[:N_displayed, 1]
                U = simu.vel[:N_displayed, 0]
                V = simu.vel[:N_displayed, 1]
                if color:
                    plt.set_cmap('hsv')
                    angle = torch.atan2(V, U)
                    quiv = ax.quiver(x.cpu(), y.cpu(), U.cpu(), V.cpu(),
                                     angle.cpu())
                    cb = ax.figure.colorbar(quiv, ax=ax,
                                            ticks=[-3.1415, 0, 3.1415],
                                            fraction=0.046, pad=0.04)
                else:
                    ax.quiver(x.cpu().numpy(), y.cpu().numpy(),
                              U.cpu().numpy(), V.cpu().numpy())
                if simu.bc == 'spherical':
                    theta = np.linspace(0, 2 * math.pi, 1000)
                    r = simu.L[0].cpu() / 2
                    ax.plot(r + r * np.cos(theta), r + r * np.sin(theta))
                ax.set_xlim(xmin=0, xmax=simu.L[0].cpu())
                ax.set_ylim(ymin=0, ymax=simu.L[1].cpu())
                ax.set_aspect('equal')
                ax.set_title(
                    simu.name + '\n Parameters: ' + simu.parameters
                    + '\n Time=' + str(round(simu.iteration * simu.dt, 1)),
                    fontsize=10)
            elif simu.d == 3:
                f = plt.figure(t, figsize=figsize)
                ax = f.add_subplot(111, projection='3d')
                X = simu.pos[:N_displayed, 0].cpu()
                Y = simu.pos[:N_displayed, 1].cpu()
                Z = simu.pos[:N_displayed, 2].cpu()
                ax.scatter(X, Y, Z, s=0.1)
                ax.set_xlim(xmin=0, xmax=simu.L[0].cpu())
                ax.set_ylim(ymin=0, ymax=simu.L[1].cpu())
                ax.set_zlim(zmin=0, zmax=simu.L[2].cpu())
                ax.set_title(
                    simu.name + '\n Parameters: ' + simu.parameters
                    + '\n Time=' + str(round(simu.iteration * simu.dt, 1)),
                    fontsize=10)
            if save:
                f.savefig(f"{final_directory}/" + str(t) + ".png")
            if show:
                # plt.show()
                plt.pause(0.000001)
            else:
                plt.close()
            t += 1
        if t > tmax:
            break

    if order:
        x = np.array(real_time)
        y = np.array(op)
        op_plot = plt.figure(t, figsize=(6, 6))
        plt.plot(x, y)
        plt.xlabel('time')
        plt.ylabel('order parameter')
        plt.axis([0, time[tmax], 0, 1])
        if save:
            simu_dir = os.path.join(current_directory, path)
            op_plot.savefig(f"{simu_dir}/order_parameter.png")
        return x, y
    else:
        return None, None


def scatter_particles(simu, time, show=True,
                      save=False, path='simu'):
    """Scatter plot with the radii of the particles.

    Args:
        simu (Particles): A model.
        time (list): List of times to plot.
        show (bool, optional): Show the plot. Default is True.
        save (bool, optional): Save the plots. Default is False.
        path (str, optional): The plots will be saved in the directory
            ``'./path/frames'``.

    """
    # Scatter plots at times specified by the list 'time' for the
    # mechanism 'mechanism' (iterator inherited from the class 'particle')
    t = 0
    tmax = len(time) - 1
    percent = 0
    real_time = [0.]
    if save:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, path, r'frames')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
    for it, info in enumerate(simu):
        real_time.append(real_time[-1] + simu.dt)
        if real_time[-1] / time[tmax] >= percent / 100:
            sys.stdout.write('\r' + "Progress:" + str(percent) + "%")
            sys.stdout.flush()
            percent += 1
        if abs(real_time[-1] >= time[t]):
            if simu.d == 2:
                f = plt.figure(t, figsize=(6, 6))
                ax = f.add_subplot(111)
                x = simu.pos[:, 0].cpu()
                y = simu.pos[:, 1].cpu()
                size = 2 * simu.R.cpu()
                offsets = list(zip(x, y))
                ax.add_collection(EllipseCollection(
                    widths=size, heights=size, angles=0, units='xy',
                    edgecolor='k', offsets=offsets, transOffset=ax.transData))
                ax.axis('equal')  # set aspect ratio to equal
                ax.axis([0, simu.L[0].cpu(), 0, simu.L[1].cpu()])
                #                 ax.set_aspect(1)
                #                 plt.axes().set_aspect('equal', 'box')
                ax.set_title(simu.name + '\n Parameters: ' + simu.parameters
                             + '\n Time=' + str(round(real_time[-1], 1)),
                             fontsize=10)
            elif simu.d == 3:
                f = plt.figure(t, figsize=(10, 10))
                ax = f.add_subplot(111, projection='3d')
                X = simu.pos[:, 0].cpu()
                Y = simu.pos[:, 1].cpu()
                Z = simu.pos[:, 2].cpu()
                ax.scatter(X, Y, Z, s=0.1)
                ax.set_xlim(0, simu.L[0].cpu())
                ax.set_ylim(0, simu.L[1].cpu())
                ax.set_zlim(0, simu.L[2].cpu())
                ax.set_title(
                    simu.name + '\n Parameters: ' + simu.parameters
                    + '\n Time=' + str(round(simu.iteration * simu.dt, 1)),
                    fontsize=10)
            if save:
                f.savefig(f"{final_directory}/" + str(t) + ".png")
            if show:
                # plt.show()
                plt.pause(0.000001)
            else:
                plt.close()
            t += 1
        if t > tmax:
            break


def heatmap_particles(mechanism, time, N_displayed, nbins, max_density=25, order=False, show=True, save=False,
                      path='simu'):
    # Scatter plots at times specified by the list 'time' for the
    # mechanism 'mechanism' (iterator inherited from the class 'particle')
    t = 0
    tmax = len(time) - 1
    percent = 0
    real_time = [0.]
    op = [mechanism.order_parameter]
    if save:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, path, r'frames')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
    for it, info in enumerate(mechanism):
        real_time.append(real_time[-1] + mechanism.dt)
        if order:
            op.append(mechanism.order_parameter)
        if real_time[-1] / time[tmax] >= percent / 100:
            sys.stdout.write('\r' + "Progress:" + str(percent) + "%")
            sys.stdout.flush()
            percent += 1
        if abs(real_time[-1] >= time[t]):
            f = plt.figure(t, figsize=(12, 6))
            ax = f.add_subplot(121)

            x = mechanism.pos[0:N_displayed, 0].cpu().numpy()
            y = mechanism.pos[0:N_displayed, 1].cpu().numpy()
            L0 = float(mechanism.L[0].cpu())
            L1 = float(mechanism.L[1].cpu())
            counts, xedges, yedges, im = ax.hist2d(x, y, bins=nbins, range=np.array([[0, L0], [0, L0]]), density=True)
            ax.axis('equal')  # set aspect ratio to equal
            ax.axis([0, L0, 0, L1])
            #                 ax.set_aspect(1)
            #                 plt.axes().set_aspect('equal', 'box')
            ax.set_title(mechanism.name + '\n Parameters: ' + mechanism.parameters
                         + '\n Time=' + str(round(real_time[-1], 1)), fontsize=10)
            im.set_clim(0, max_density)
            cb = f.colorbar(im, ax=ax)

            ax1 = f.add_subplot(122)
            #             U = mechanism.vel[0:N_displayed,0].cpu()
            #             V = mechanism.vel[0:N_displayed,1].cpu()
            #             ax1.quiver(x,y,U,V)
            #             ax1.axis('equal')
            #             ax1.axis([0,L0,0,L1])

            if save:
                f.savefig(f"{final_directory}/" + str(t) + ".png")
            if show:
                plt.show()
            else:
                plt.close()
            t += 1
        if t > tmax:
            break
    if order:
        x = np.array(real_time)
        y = np.array(op)
        op_plot = plt.figure(t, figsize=(6, 6))
        plt.plot(x, y)
        plt.xlabel('time')
        plt.ylabel('order parameter')
        plt.axis([0, time[tmax], 0, 1])
        if save:
            simu_dir = os.path.join(current_directory, path)
            op_plot.savefig(f"{simu_dir}/order_parameter.png")
        return x, y

def save_bo(mechanism, to_save, Nmax=2000, order=False, path='data'):
    N_saved = min(Nmax, mechanism.N)
    percent = 0
    theta = []
    phi = []
    real_time = []
    should_save = False
    savei = 0
    max_time = to_save[-1]
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r"" + path)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for it, info in enumerate(mechanism):
        real_time.append(it * mechanism.dt)
        theta.append(mechanism.theta)
        phi.append(mechanism.phi)

        if it == np.floor((percent / 100) * max_time / mechanism.dt):
            sys.stdout.write('\r' + "Progress:" + str(percent) + "%")
            sys.stdout.flush()
            percent += 1
        if it * mechanism.dt > max_time:
            break

        should_save = it * mechanism.dt >= to_save[savei]
        if should_save:
            torch.save(mechanism.pos[range(N_saved), :], final_directory + '/pos' + str(savei) + '.pt')
            torch.save(mechanism.bo[range(N_saved), :], final_directory + '/bo' + str(savei) + '.pt')
            savei += 1
    x = np.array(real_time)
    y = np.array(theta)
    z = np.array(phi)
    theta_plot = plt.figure(1, figsize=(6, 6))
    plt.plot(x, y)
    plt.xlabel('time')
    plt.ylabel('theta')
    plt.axis([0, max_time, 0, 3.15])
    theta_plot.savefig(f"{final_directory}/theta.png")
    phi_plot = plt.figure(2, figsize=(6, 6))
    plt.plot(x, z)
    plt.xlabel('time')
    plt.ylabel('phi')
    plt.axis([0, max_time, 0, 6.3])
    theta_plot.savefig(f"{final_directory}/phi.png")
    param = open(final_directory + '/parameters.txt', 'w')
    param.write(mechanism.parameters)
    param.close()
    times = open(final_directory + '/times_saved.txt', 'w')
    times.write(str(to_save))
    times.close()
    np.save(final_directory + '/times.npy', x)
    np.save(final_directory + '/theta.npy', y)
    np.save(final_directory + '/phi.npy', z)
    return real_time, theta, phi


def plot_bo_twist_order(mechanism,
                        to_plot,
                        K,
                        axis,
                        Nmax=2000,
                        show=True,
                        save_particles=False,
                        save_twist=False,
                        save_order=False,
                        path='data'):
    N_saved = min(Nmax, mechanism.N)
    percent = 0
    theta = []
    phi = []
    real_time = []
    local_order = []
    global_order = []
    ordertime = []
    should_plot = False
    ploti = 0
    which_axis = len(axis)
    colors = range(K)
    max_time = to_plot[-1]
    slices = torch.zeros((K, 2, len(to_plot), len(axis))).type_as(mechanism.pos)
    if save_particles or save_twist or save_order:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, r"" + path)
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        if save_particles:
            os.makedirs(os.path.join(final_directory, r'particles'))
        if save_twist:
            os.makedirs(os.path.join(final_directory, r'twist'))
    for it, info in enumerate(mechanism):
        real_time.append(it * mechanism.dt)
        theta.append(mechanism.theta)
        phi.append(mechanism.phi)

        if it == np.floor((percent / 100) * max_time / mechanism.dt):
            sys.stdout.write('\r' + "Progress:" + str(percent) + "%")
            sys.stdout.flush()
            percent += 1
        if it * mechanism.dt > max_time:
            break

        should_plot = it * mechanism.dt >= to_plot[ploti]
        if should_plot:
            if save_particles:
                torch.save(mechanism.pos[range(N_saved), :], final_directory + '/particles/pos' + str(ploti) + '.pt')
                torch.save(mechanism.bo[range(N_saved), :], final_directory + '/particles/bo' + str(ploti) + '.pt')
            if save_order:
                ordertime.append(it * mechanism.dt)
                local_order.append(mechanism.local_order)
                global_order.append(mechanism.global_order)
            for ax in range(which_axis):
                slice_p, slice_q = mechanism.twist(K, axis[ax])
                slices[:, :, ploti, ax] = torch.cat((slice_p.reshape((K, 1)), slice_q.reshape((K, 1))), axis=1)
                slice_p = slice_p.cpu()
                slice_q = slice_q.cpu()
                axf, axi = plt.subplots(figsize=(10, 9))
                angle = np.arange(0., 2 * np.pi, 0.01)
                axi.plot(np.cos(angle), np.sin(angle), linewidth=6)
                center = axi.scatter([0.], [0.], s=200, marker='X')
                scat = axi.scatter(slice_p, slice_q, c=colors, cmap='hsv')

                axi.set_xlim(-1.1, 1.1)
                axi.set_xticks([-1, 0, 1])
                axi.set_xticklabels([-1, 0, 1], fontsize=48)
                axi.set_ylim(-1.1, 1.1)
                axi.set_yticks([-1, 0, 1])
                axi.set_yticklabels([-1, 0, 1], fontsize=48)
                axi.set_aspect('equal', 'box')
                axi.set_title('Twist along the axis ' + axis[ax] + ' \n Time=' + str(round(to_plot[ploti], 1)),
                              fontsize=36)
                cb = axi.figure.colorbar(scat, ax=axi, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=36)
                if show:
                    plt.show()
                else:
                    plt.close()
                if save_twist:
                    axf.savefig(f"{final_directory}/twist/twist_" + axis[ax] + "_" + str(ploti) + ".png")
            ploti += 1
    if save_twist:
        torch.save(slices, final_directory + '/slices.pt')
    x = np.array(real_time)
    y = np.array(theta)
    z = np.array(phi)
    theta_plot = plt.figure(1, figsize=(12, 12))
    plt.plot(x, y, linewidth=12)
    plt.xlabel('time', fontsize=36)
    plt.xticks(ticks=[0, max_time / 4, max_time / 2, 3 * max_time / 4, round(max_time)],
               labels=[0, round(max_time / 4), round(max_time / 2), round(3 * max_time / 4), max_time], fontsize=36)
    plt.ylabel('θ', fontsize=36, rotation=0, labelpad=20)
    plt.yticks(fontsize=36)
    plt.yticks(ticks=[0, np.pi / 2, np.pi], labels=['0', 'π/2', 'π'], fontsize=36)
    plt.hlines([0, np.pi / 2, np.pi], xmin=0, xmax=max_time, linestyles='dashed', linewidth=4)
    plt.axis([0, max_time, 0, 3.15])

    phi_plot = plt.figure(2, figsize=(12, 12))
    plt.plot(x, z, linewidth=12)
    plt.xlabel('time', fontsize=36)
    plt.xticks(ticks=[0, max_time / 4, max_time / 2, 3 * max_time / 4, max_time],
               labels=[0, round(max_time / 4), round(max_time / 2), round(3 * max_time / 4), round(max_time)],
               fontsize=36)
    plt.ylabel('ᵠ', fontsize=52, rotation=0, labelpad=10)
    plt.yticks(ticks=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], labels=['0', 'π/2', 'π', '3π/2', '2π'],
               fontsize=36)
    plt.hlines([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], xmin=0, xmax=max_time, linestyles='dashed', linewidth=6)
    plt.axis([0, max_time, 0, 6.3])
    if save_order:
        ordertime = np.array(ordertime)
        np.save(final_directory + '/order_times.npy', ordertime)

        local_order = np.array(local_order)
        locop_plot = plt.figure(3, figsize=(8, 8))
        plt.plot(ordertime, local_order, linewidth=6)
        plt.xlabel('time', fontsize=36)
        plt.ylabel('order parameter', fontsize=36)
        locop_plot.savefig(f"{final_directory}/local_order_parameter.png")
        np.save(final_directory + '/local_order_parameter.npy', local_order)

        global_order = np.array(global_order)
        globop_plot = plt.figure(4, figsize=(18, 18))
        plt.plot(ordertime, global_order, linewidth=12)
        plt.xlabel('time', fontsize=48)
        plt.xticks(ticks=[0, max_time / 4, max_time / 2, 3 * max_time / 4, max_time],
                   labels=[0, round(max_time / 4), round(max_time / 2), round(3 * max_time / 4), round(max_time)],
                   fontsize=48)
        plt.ylabel('order parameter', fontsize=48)
        plt.yticks(ticks=[0, 0.25, 0.451, 0.854, 1], labels=['0', '0.25', '0.45', '0.85', '1'], fontsize=48)
        plt.hlines([0.25, 0.451, 0.854], xmin=0, xmax=max_time, linestyles='dashed', linewidth=6)
        plt.axis([0, max_time, 0, 1.1])
        globop_plot.savefig(f"{final_directory}/global_order_parameter.png")
        np.save(final_directory + '/global_order_parameter.npy', global_order)

    if save_particles or save_twist:
        theta_plot.savefig(f"{final_directory}/theta.png")
        phi_plot.savefig(f"{final_directory}/phi.png")
        param = open(final_directory + '/parameters.txt', 'w')
        param.write(mechanism.parameters)
        param.close()
        times = open(final_directory + '/times_saved.txt', 'w')
        times.write(str(to_plot))
        times.close()
        np.save(final_directory + '/times.npy', x)
        np.save(final_directory + '/theta.npy', y)
        np.save(final_directory + '/phi.npy', z)
    return real_time, theta, phi, slices
