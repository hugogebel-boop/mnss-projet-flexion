#!/usr/bin/env python3
"""
Module: plot.py
Description: Contains plotting functions and mesh‐related functions.
"""

import matplotlib
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import meshio
import subprocess
import os
from matplotlib import collections as mc
from IPython.display import display
from sympy import *

# Import FEM shape functions for interpolation
from fem_utils import N1, N2, N3, N4, assemblerMatriceRigidite

# Import utility functions for geometric calculations and color conversion
from utils import convert_colors, compute_minmax, compute_range, compute_node_size

# For animation display (adjust as needed)
from Slides import presentation_helper as ph

# Configure matplotlib parameters
matplotlib.rcParams["figure.figsize"] = (9, 6)
matplotlib.rcParams.update({"font.size": 22})
matplotlib.rcParams.update({"legend.fontsize": 22})
matplotlib.rcParams.update({"lines.linewidth": 4})
matplotlib.rcParams.update({"lines.markersize": 10})
matplotlib.rcParams.update({"axes.labelsize": 30})


def plot_structure(
    positions,
    conn,
    plot_eqn=None,
    elem_colors=None,
    node_colors=None,
    show_nodes=True,
    show_node_indexes=True,
    show_elems=True,
    show_elem_indexes=True,
    linestyle="-",
    ax=None,
    **kwargs,
):
    """
    Plots a structure defined by node positions and connectivity.
    Optionally displays equation numbering.
    """
    positions = np.array(positions)
    conn = np.array(conn)
    eqn_num_node = None

    # Determine equation numbering strategy if provided
    if isinstance(plot_eqn, str):
        from fem_utils import packed_eqn, stride_eqn, eqn_number_node

        if plot_eqn == "packed":
            plot_eqn = packed_eqn
        elif plot_eqn == "stride":
            plot_eqn = stride_eqn
        else:
            raise RuntimeError("Unknown eqn packing strategy")
        eqn_num_node = eqn_number_node(plot_eqn, positions.shape[0])
    elif isinstance(plot_eqn, np.ndarray):
        eqn_num_node = plot_eqn
    elif plot_eqn is not None:
        raise RuntimeError(
            "Could not interpret plot_eqn of type " + str(type(plot_eqn))
        )

    if eqn_num_node is not None:
        from fem_utils import eqn_number_elem

        eqn_num_elem = eqn_number_elem(eqn_num_node, conn)

    _min, _max = compute_minmax(positions)
    _range = compute_range(positions)
    # Adjust range if zero in any direction
    for i in range(2):
        if _range[i] == 0:
            _range[i] = 0.5

    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim((_min[0] - _range[0] * 0.2, _max[0] + _range[0] * 0.2))
    ax.set_ylim((_min[1] - _range[1] * 0.2, _max[1] + _range[1] * 0.2))

    lines = []
    for e in conn:
        p1 = positions[e[0]]
        p2 = positions[e[1]]
        lines.append((p1, p2))

    elem_colors = convert_colors(elem_colors)
    node_colors = convert_colors(node_colors)

    if show_elems:
        lc = mc.LineCollection(
            lines, linewidths=2, colors=elem_colors, linestyle=linestyle
        )
        ax.add_collection(lc)

    if show_nodes:
        disp_positions = ax.transData.transform(positions)
        node_size_px = compute_node_size(disp_positions)
        ax.scatter(positions[:, 0], positions[:, 1], s=node_size_px**2, c=node_colors)

    node_size = compute_node_size(positions)
    center_gravity = (_max + _min) / 2

    for i, p in enumerate(positions):
        _n = p - center_gravity
        norm = np.linalg.norm(_n)
        if norm < 1e-5:
            _n = np.array([1, 1])
        norm = _range.max() * 0.08 / np.linalg.norm(_n)
        _n *= norm
        pos = p
        if show_nodes and show_node_indexes:
            ax.text(
                pos[0],
                pos[1],
                str(i),
                horizontalalignment="center",
                verticalalignment="center",
            )
        if eqn_num_node is not None:
            eqns = eqn_num_node[i, :]
            ax.text(
                pos[0] + _n[0] * 1.5,
                pos[1] + _n[1] * 1.5,
                "[" + ",".join([str(int(e)) for e in eqns]) + "]",
                horizontalalignment="center",
                verticalalignment="center",
            )

    for i, e in enumerate(conn):
        p1 = positions[e[0]]
        p2 = positions[e[1]]
        center = (p1 + p2) / 2
        _l = np.zeros(3)
        _l[:2] = p2 - p1
        _l /= np.linalg.norm(_l)
        _n = np.cross(_l, [0, 0, 1])
        center += _n[:2] * node_size * 1.5
        if show_elems and show_elem_indexes:
            ax.text(
                center[0],
                center[1],
                f"({i})",
                horizontalalignment="center",
                verticalalignment="center",
            )

    ret = {}
    if eqn_num_node is not None:
        ret["eqn_node"] = np.array(eqn_num_node, dtype=int)
        from fem_utils import eqn_number_elem

        ret["eqn_elem"] = np.array(eqn_number_elem(eqn_num_node, conn), dtype=int)
    else:
        ret = None

    return ret


def plotMesh(
    coords,
    connectivity,
    nodal_field=None,
    elemental_field=None,
    colormap="viridis",
    **kwargs,
):
    """
    Visualizes 2D finite element meshes (T3 linear or T6 quadratic triangles) with contour plots.

    Note on T6 Elements:
    Standard plotting libraries only support 3-node triangles. T6 elements are dynamically
    subdivided into four smaller T3 sub-triangles for visualization. This means the drawn
    mesh lines will appear denser than your actual element count, but the physical geometry
    is accurately preserved.

    Parameters:
    -----------
    coords : numpy.ndarray
        Node coordinates array of shape (N_nodes, 2).
    connectivity : numpy.ndarray
        Element definitions mapping elements to node indices.
        Shape (N_elements, 3) for T3 or (N_elements, 6) for T6.
    nodal_field : numpy.ndarray, optional
        Continuous data evaluated at the nodes (e.g., displacements, smoothed stresses).
        Expected shape is (N_nodes, N_fields) or a flattened array of size N_nodes * N_fields.
    elemental_field : numpy.ndarray, optional
        Discrete data evaluated per element (e.g., raw element stresses/strains).
        Expected shape is (N_elements, N_fields) or a flattened array of size N_elements * N_fields.
    colormap : str, default="viridis"
        Matplotlib colormap name. Tip: Use "coolwarm" or "RdBu_r" for zero-centered
        diverging data like tension/compression.
    **kwargs : dict
        Additional arguments passed to `generate_scalar_field()` (e.g., `axis=0` to
        extract the X-component from a vector field).
    """
    nodes_per_elem = connectivity.shape[1]

    # Handle Connectivity (Subdivide T6 to 4xT3 for plotting)
    if nodes_per_elem == 3:
        plot_conn = connectivity
    elif nodes_per_elem == 6:
        # GMSH T6 convention: 0,1,2 (corners), 3 (mid 0-1), 4 (mid 1-2), 5 (mid 2-0)
        tri1 = connectivity[:, [0, 3, 5]]
        tri2 = connectivity[:, [3, 1, 4]]
        tri3 = connectivity[:, [5, 4, 2]]
        tri4 = connectivity[:, [3, 4, 5]]
        # Stack into a (4 * N_elem, 3) array
        plot_conn = np.vstack([tri1, tri2, tri3, tri4])
        # Print a clearer warning about the visual density of T6 meshes
        print(
            "Note: T6 elements are subdivided into 4 T3 sub-triangles for plotting. "
            "Mesh lines will appear denser than the actual element count."
        )
    else:
        raise ValueError(
            f"Unsupported number of nodes per element: {nodes_per_elem}. Expected 3 or 6."
        )

    triangles = tri.Triangulation(coords[:, 0], coords[:, 1], plot_conn)
    plt.gca().set_aspect("equal")

    # Plot Nodal Field
    if nodal_field is not None:
        n_field = nodal_field.reshape(
            coords.shape[0], nodal_field.size // coords.shape[0]
        )
        n_field = generate_scalar_field(n_field, **kwargs)
        contour = plt.tricontourf(triangles, n_field, cmap=colormap)
        plt.colorbar(contour)

    # Plot Elemental Field
    if elemental_field is not None:
        e_field = elemental_field.reshape(
            connectivity.shape[0], elemental_field.size // connectivity.shape[0]
        )
        e_field = generate_scalar_field(e_field, **kwargs)

        if nodes_per_elem == 6:
            # Duplicate each element's scalar value 4 times for the 4 sub-triangles
            e_field = np.concatenate([e_field, e_field, e_field, e_field])

        contour = plt.tripcolor(triangles, e_field, cmap=colormap)
        plt.colorbar(contour)

    # Draw Mesh Lines
    plt.triplot(triangles, "--", color="black", lw=0.5, alpha=0.6)


def readMesh(filename, element_types=["triangle", "triangle6"]):
    """
    Reads a mesh file and returns the coordinates and connectivity.
    Now defaults to catching both T3 and T6 elements.
    """
    mesh = meshio.read(filename)

    if isinstance(element_types, str):
        element_types = [element_types]

    for c in mesh.cells:
        if c.type in element_types:
            return mesh.points[:, :2], np.array(c.data)

    print(f"Warning: No elements of type {element_types} found.")
    return None


def meshGeo(filename, dim=2, order=1):
    """
    Generates a mesh using gmsh and returns the mesh read from the generated file.
    Order=1 yields T3, Order=2 yields T6.
    """
    ret = subprocess.run(
        f"gmsh -{dim} -order {order} -o tmp.msh {filename}",
        shell=True,
        capture_output=True,
    )
    if ret.returncode:
        print("Beware, gmsh could not run: mesh is not generated")
        print(ret.stderr.decode())  # Added error output for easier debugging
        return None
    else:
        print(f"Mesh generated (Order {order})")
        # Direct readMesh to look for the correct element type first
        target_type = "triangle6" if order == 2 else "triangle"
        mesh = readMesh("tmp.msh", element_types=[target_type, "triangle"])

        if os.path.exists("tmp.msh"):
            os.remove("tmp.msh")
        return mesh


def spring_animation(nodes, disp, xlim=None, ylim=(-0.5, 0.5)):
    """
    Creates an animation of spring displacement over time.
    """
    nsteps = disp.shape[1]
    if xlim is None:
        xlim = (nodes.min(), nodes.max())
    fig = plt.figure()
    ax = plt.axes(xlim=xlim, ylim=ylim)
    (line,) = ax.plot([], [], "-")

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(nodes, disp[i, :])
        return (line,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=int(nsteps), interval=20, blit=True
    )
    return ph.display_animation(anim)


def create_element_lines(p1, p2, u1=None, u2=None, n=20, interpolate=None):
    """
    Generates a list of line segments representing an element between points p1 and p2,
    optionally incorporating displacement data.
    """
    L = np.linalg.norm(p2 - p1)
    e1 = np.zeros(3)
    e3 = np.zeros(3)
    e3[2] = 1.0
    e1[:2] = (p2 - p1) / L
    e2 = np.cross(e3, e1)
    xi = np.linspace(0, L, n + 1)
    # Compute interpolation using FEM shape functions imported from fem_utils
    X = np.outer(N1(L, xi), p1) + np.outer(N3(L, xi), p2)
    if (u1 is not None) and (u2 is not None):
        X += np.outer(N1(L, xi), u1[:2]) + np.outer(N3(L, xi), u2[:2])
        X += np.outer(N2(L, xi), e2[:2] * u1[2]) + np.outer(N4(L, xi), e2[:2] * u2[2])
    res = [(X[i], X[i + 1]) for i in range(len(X) - 1)]
    return res


def plot_elements(
    ax,
    positions,
    conn,
    displacement=None,
    linestyle=None,
    elem_colors=None,
    n=20,
    show_number=True,
    **kwargs,
):
    """
    Plots finite element lines based on node positions, connectivity, and optional displacement.
    """
    lines = []
    for i, e in enumerate(conn):
        p1 = positions[e[0]]
        p2 = positions[e[1]]
        u1 = u2 = None
        if displacement is not None:
            u1 = displacement[e[0]]
            u2 = displacement[e[1]]
        lines += create_element_lines(p1, p2, n=n, u1=u1, u2=u2)
    elem_colors = convert_colors(elem_colors)
    if linestyle is None:
        linestyle = "-"
    lc = mc.LineCollection(
        lines, linewidths=2, linestyles=linestyle, colors=elem_colors
    )
    ax.add_collection(lc)
    node_size = compute_node_size(positions)
    if show_number:
        for i, e in enumerate(conn):
            p1 = positions[e[0]]
            p2 = positions[e[1]]
            center = (p1 + p2) / 2
            _l = np.zeros(3)
            _l[:2] = p2 - p1
            _l /= np.linalg.norm(_l)
            _n = np.cross(_l, [0, 0, 1])
            center += _n[:2] * node_size * 1.5
            ax.text(
                center[0],
                center[1],
                f"({i})",
                horizontalalignment="center",
                verticalalignment="center",
            )


def plot_bloc_stiffness(positions, conn, E, A, nb_barre):

    materiau = E * A * np.ones(conn.shape[0], dtype=object)
    elem_colors = np.array(["b", "g", "r", "y"])

    for i in range(nb_barre):
        colors = np.ones(conn.shape[0], dtype=str)
        colors[:] = "k"
        colors[i] = elem_colors[i]
        plot_structure(positions, conn, elem_colors=colors)

        K = assemblerMatriceRigidite(
            positions,
            conn,
            materiau,
            elem_colors=elem_colors,
            elem_to_assemble=range(i + 1),
        )
        K.evalf(5)
        display(K.profile(remove_zeros=True))
        # display(K.profile())


def generate_scalar_field(field, axis=None):
    """
    Generates a scalar field from a vector field.
    """
    if axis is None:
        return np.linalg.norm(field, axis=1)
    else:
        return field[:, axis]


def modesAnimation(nodes, xlim=None, ylim=(-0.5, 0.5), eigs=None, marker=None):

    eigen_values = eigs[0].real
    eigen_vectors = eigs[1]

    if nodes.shape[0] != eigen_vectors[:, 0].shape[0]:
        nodes = nodes[1:-1]

    eigen_values[eigen_values < 0] = 0
    omegas = np.sqrt(eigen_values)
    T = 2.0 * np.pi / omegas[0]
    if np.abs(omegas[0]) < 1e-12:
        T = 2.0 * np.pi / omegas[1]

    n_modes = eigen_values.shape[0]

    nsteps = 200 * n_modes
    time_factor = 2 * T / nsteps

    if xlim is None:
        xlim = (nodes.min(), nodes.max())

    fig = plt.figure()
    fig.subplots_adjust(left=0.16, right=0.95, top=0.95, bottom=0.15)
    ax = plt.axes(xlim=xlim, ylim=ylim)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Deplacement $d(t)$")
    lines = [
        ax.plot(nodes, eigen_vectors[:, 0].real, marker=marker, lw=2)[0]
        for i in range(0, n_modes)
    ]

    def init():
        for i in range(0, n_modes):
            lines[i].set_data([], [])
        return lines

    def animate(ts):
        nn_modes = 1 + int(ts / 200)
        if nn_modes > n_modes:
            nn_modes = n_modes

        for i in range(0, nn_modes):
            lines[i].set_data(
                nodes, eigen_vectors[:, i].real * np.cos(omegas[i] * ts * time_factor)
            )

        return lines

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=nsteps, interval=10, blit=True
    )

    return anim
