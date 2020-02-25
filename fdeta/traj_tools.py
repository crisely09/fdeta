# -*- coding: utf-8 -*-
#  by CGE, 2020.
"""
Tools for MDTrajectory Class.

"""

import numpy as np
from typing import Union
from qcelemental import periodictable


def compute_center_of_mass(mass: np.ndarray, coordinates: np.ndarray) -> float:
    return np.sum(np.multiply(coordinates, mass.reshape(len(mass), 1)), axis=0)/mass.sum()


def atom_to_mass(atom: Union[str, int]) -> float:
    """Give back the mass of a given element."""
    return periodictable.to_mass(atom)


def atom_to_charge(atom: Union[str, int]) -> float:
    """Give the nucleus atomic charge."""
    return float(periodictable.to_atomic_number(atom))


def default_charges(elements: list) -> dict:
    """Get the default charges for a list of given elements."""
    charges = {}
    for i, atom in enumerate(elements):
        charges[atom] = (i, atom_to_charge(atom))
    return charges
