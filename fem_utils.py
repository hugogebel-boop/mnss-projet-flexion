#!/usr/bin/env python3
"""
Module: fem_utils.py
Description: Contains FEM computations, energy analysis, and simulation functions.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from sympy import Matrix, simplify
from IPython.display import display
from Slides import math_helper as mh  # For matrix printing and colored matrices
import matplotlib.pyplot as plt
import scipy


def packed_eqn(node_idx, dof_per_node, number_of_nodes):
    """
    Equation numbering using packed strategy.
    """
    return np.arange(dof_per_node) + np.ones(dof_per_node) * node_idx * dof_per_node


def stride_eqn(node_idx, dof_per_node, number_of_nodes):
    """
    Equation numbering using stride strategy.
    """
    return np.arange(dof_per_node) * number_of_nodes + node_idx


def eqn_number_node(f_node_eqn, number_of_nodes):
    """
    Generates equation numbers for nodes.
    """
    eqn_number = []
    for n in range(number_of_nodes):
        eqn = f_node_eqn(n, 2, number_of_nodes)
        eqn_number.append(np.array(eqn).flatten())
    return np.array(eqn_number, dtype=int)


def eqn_number_elem(eqn_number_node, conn):
    """
    Generates equation numbers for elements based on node numbering.
    """
    eqn_number = []
    for e in conn:
        eqn = []
        eqn.append(eqn_number_node[e[0], :])
        eqn.append(eqn_number_node[e[1], :])
        eqn_number.append(np.array(eqn).flatten())
    return np.array(eqn_number, dtype=int)


def calculerMatriceRotation(p1, p2):
    """
    Computes the rotation matrix from local to global coordinates and the element length.
    """
    barre = Matrix(p2 - p1)
    L = barre.norm()
    R = 1 / L * np.array([[barre[0], -barre[1]], [barre[1], barre[0]]])
    return simplify(Matrix(R)), L


def calculerMatriceRigiditeLocale(k):
    """
    Computes the local stiffness matrix.
    """
    Kl = k * Matrix([[1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]])
    return simplify(Kl)


def assemblerMatriceRigidite(
    coordonnees,
    connectivites,
    materiau,
    elem_colors=None,
    elem_to_assemble=None,
    **kwargs
):
    """
    Assembles the global stiffness matrix for a finite element system.
    """
    eqn_num_node = eqn_number_node(packed_eqn, coordonnees.shape[0])
    equations_num = eqn_number_elem(eqn_num_node, connectivites)
    nb_noeuds = coordonnees.shape[0]
    nb_elem = connectivites.shape[0]
    nb_noeuds_p_elem = connectivites.shape[1]
    nb_ddl_p_noeuds = 2
    K = np.zeros(
        (nb_noeuds * nb_ddl_p_noeuds, nb_noeuds * nb_ddl_p_noeuds), dtype=object
    )
    if elem_to_assemble is None:
        elem_to_assemble = range(nb_elem)
    for e in elem_to_assemble:
        n_1 = coordonnees[connectivites[e, 0], :]
        n_2 = coordonnees[connectivites[e, 1], :]
        R, L = calculerMatriceRotation(n_1, n_2, **kwargs)
        T = np.zeros(
            (nb_noeuds_p_elem * nb_ddl_p_noeuds, nb_noeuds_p_elem * nb_ddl_p_noeuds),
            dtype=object,
        )
        T[:2, :2] = R
        T[2:, 2:] = R
        Kl = calculerMatriceRigiditeLocale(materiau[e] / L)
        Kg = T @ Kl @ T.T
        idx = equations_num[e, :]
        for i, gi in enumerate(idx):
            for j, gj in enumerate(idx):
                K[gi, gj] += Kg[i, j]
    K = simplify(Matrix(K))
    if elem_colors is None:
        return K
    Kcolor = np.zeros_like(K, dtype="S10")
    for e in elem_to_assemble:
        idx = equations_num[e, :]
        for i, gi in enumerate(idx):
            for j, gj in enumerate(idx):
                c1 = Kcolor[gi, gj].decode()
                c2 = elem_colors[e]
                if c1 != c2:
                    c3 = c1 + c2
                else:
                    c3 = c1
                Kcolor[gi, gj] = c3.encode()
    K = mh.ColoredMatrix(K)
    K.colors = Kcolor
    return K


def computeEnergyEvolution(displacements, velocities, computeEnergy):
    """
    Computes the evolution of potential, kinetic, and total energy over time.
    """
    epot = []
    ekin = []
    etot = []
    nsteps = displacements.shape[0]
    for i in range(nsteps):
        _epot, _ekin = computeEnergy(displacements[i], velocities[i])
        _etot = _epot + _ekin
        epot.append(_epot)
        ekin.append(_ekin)
        etot.append(_etot)
    return np.array(epot), np.array(ekin), np.array(etot)


def plotEnergyEvolution(
    displacements, velocities, computeEnergy, loglog=False, semilog=False, **kwargs
):
    """
    Plots the evolution of potential, kinetic, and total energy.
    """
    nsteps = displacements.shape[0]
    epot, ekin, etot = computeEnergyEvolution(displacements, velocities, computeEnergy)
    T = range(nsteps)
    if semilog:
        plt.semilogy(T, etot, label="$E^{Tot}$", **kwargs)
    elif loglog:
        plt.loglog(T, etot, label="$E^{Tot}$", **kwargs)
    else:
        plt.plot(T, epot, label="$E^{pot}$", **kwargs)
        plt.plot(T, ekin, label="$E^{cin}$", **kwargs)
        plt.plot(T, etot, label="$E^{Tot}$", **kwargs)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", **kwargs)
    plt.ylabel("Energy [J]", **kwargs)
    plt.xlabel("Time frame [iterations]", **kwargs)


def NewMarkIntegrationBeta025Gamma05(U, V, A, M, K, dt):
    """
    Newmark integration method with beta=0.25 and gamma=0.5.
    """
    Fprime = M @ (U + dt * V + dt**2 / 4.0 * A)
    Mprime = M + dt**2 / 4 * K
    Mprime = scipy.sparse.csr_matrix(Mprime)
    U[:] = scipy.sparse.linalg.spsolve(Mprime, Fprime)
    V += dt / 2 * A
    Fprime = -K @ U
    Mprime = scipy.sparse.csr_matrix(M)
    new_A = scipy.sparse.linalg.spsolve(Mprime, Fprime)
    A[:] = new_A[:]
    V += dt / 2 * A


def makeEvolution(
    nodes,
    nsteps,
    dt=1.0,
    U=None,
    K=None,
    M=None,
    time_integration=NewMarkIntegrationBeta025Gamma05,
):
    """
    Evolves the system over time using a specified time integration method.
    Returns displacements, velocities, and forces.
    """
    displacements = []
    forces = []
    velocities = []
    V = np.zeros_like(nodes)
    A = np.zeros_like(nodes)
    if callable(U):
        U = U(nodes)
    if callable(K):
        K = K(nodes.shape[0])
    if callable(M):
        M = M(nodes.shape[0])
    K = scipy.sparse.csr_matrix(K)
    M = scipy.sparse.csr_matrix(M)
    for s in range(nsteps):
        displacements.append(U.copy())
        velocities.append(V.copy())
        forces.append((K @ U).copy())
        if s % 100 == 0:
            print(s)
        time_integration(U, V, A, M, K, dt)
    return np.array(displacements), np.array(velocities), np.array(forces)


def N1(L, xi):
    """
    Shape function N1.
    """
    return 1 / L**3 * (2 * xi**3 - 3 * xi**2 * L + L**3)


def N2(L, xi):
    """
    Shape function N2.
    """
    return 1 / L**3 * (xi**3 * L - 2 * xi**2 * L**2 + xi * L**3)


def N3(L, xi):
    """
    Shape function N3.
    """
    return 1 / L**3 * (-2 * xi**3 + 3 * xi**2 * L)


def N4(L, xi):
    """
    Shape function N4.
    """
    return 1 / L**3 * (xi**3 * L - xi**2 * L**2)
