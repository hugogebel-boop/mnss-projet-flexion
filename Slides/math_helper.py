"""Stub de Slides.math_helper — affichage LaTeX dans Jupyter, sans dépendance wand."""
import numpy as np
from sympy import latex, Matrix
from IPython.display import display, Math


def print_latex(template, *args, ret=False):
    """Reproduit la signature : `print_latex("K = {0}", K)` -> rendu LaTeX."""
    rendered = [latex(a) if not isinstance(a, str) else a for a in args]
    s = template.format(*rendered)
    if ret:
        return s
    display(Math(s))


class ColoredMatrix:
    """Stub : on garde la matrice sympy/numpy, on ignore la coloration."""

    def __init__(self, matrix):
        if isinstance(matrix, np.ndarray):
            matrix = Matrix(matrix)
        self.matrix = matrix
        self.colors = None

    def __getattr__(self, name):
        return getattr(self.matrix, name)

    def profile(self, remove_zeros=False):
        return self.matrix

    def evalf(self, n):
        return self.matrix.evalf(n)

    def _repr_latex_(self):
        return f"$${latex(self.matrix)}$$"
