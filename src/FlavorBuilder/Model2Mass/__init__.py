"""
Model2Mass Subpackage
=====================

This subpackage provides tools for constructing and analyzing
particle-physics models based on mass matrices.

It is designed to interface seamlessly with the `PyDiscrete`
subpackage, which supplies finite group–theory utilities
via the GAP system.

"""

# Import main user-facing API from the module
from .Model2Mass import Model2Mass, make_latex_tableA4, make_latex_tableT19

__all__ = ["Model2Mass", "make_latex_tableA4", "make_latex_tableT19"]
__version__ = "0.1.0"
