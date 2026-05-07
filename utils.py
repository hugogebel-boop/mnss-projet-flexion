#!/usr/bin/env python3
"""
Module: utils.py
Description: Contains utility functions for general use.
"""

import numpy as np
import matplotlib.colors as pltcol
from IPython.display import IFrame, display
from Slides import math_helper as mh
from sympy import *


def plot_matrix(matrix, matrix_name):
    """
    Prints the matrix in LaTeX format using math_helper.
    """
    if isinstance(matrix, np.ndarray):
        matrix = Matrix(matrix)
    if isinstance(matrix, list):
        matrix = Matrix(matrix)
    mh.print_latex(matrix_name + "= {0}", matrix)


def plot_matrix_product(matrix1, matrix2, matrix3, matrix_name):
    """
    Prints the product of matrices in LaTeX format using math_helper.
    """
    mh.print_latex(matrix_name + "= {0}{1}{2}", matrix1, matrix2, matrix3)


def compute_minmax(positions):
    """
    Computes the minimum and maximum coordinates from positions.
    """
    return np.array(positions.min(axis=0), dtype=float), np.array(
        positions.max(axis=0), dtype=float
    )


def compute_range(positions):
    """
    Computes the range (max - min) of the positions.
    """
    _min, _max = compute_minmax(positions)
    return _max - _min


def compute_node_size(positions):
    """
    Computes a node size for plotting based on the positions.
    """
    _range = compute_range(positions)
    max_range = _range.max()
    return max_range * 0.08


def convert_colors(cols, default_color="c"):
    """
    Converts a list of color strings to RGB tuples.
    """
    if cols is None:
        return cols
    for i in range(len(cols)):
        if cols[i] == "":
            cols[i] = default_color
    return [pltcol.to_rgb(c) for c in cols]


def votre_opinion_compte(name):
    """
    Displays an IFrame for a survey.
    """
    url = f"https://www.surveymonkey.com/r/NOTOSURVEY?notebook_set=CIVIL-321&notebook_id=CIVIL-321{name}"
    display(IFrame(url, width=600, height=1000))
