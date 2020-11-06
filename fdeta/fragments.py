# -*- coding: utf-8 -*-
#  by CGE, 2020.
"""
Tools for Fragment/Molecule handling.

"""

import os
import numpy as np


def get_bond_matrix(elements, geos, unit='bohr'):
    """Make a bond matrix from an array of geometries.

    Parameters
    ----------
    geos : np.ndarray((N, 3))
        Geometries of all atoms.

    Returns
    -------
    bond_matrix : np.ndarray((NxN))
        Bond matrix
    """
    if len(elements) != len(geos):
        raise ValueError("Number of lements and geos don't match.")
    if unit == 'angstrom':
        geos /= BOHR
    elif unit != 'bohr':
        raise ValueError('`unit` can only be `bohr` or `angstrom`')
    natoms = geos.shape[0]
    bond_matrix = np.zeros((natoms, natoms), dtype=int)
    for i in range(natoms):
        for j in range(natoms):
            if i != j:
                d = np.linalg.norm(geos[i] - geos[j])
                limit = qce.vdwradii.get(elements[i]) + qce.vdwradii.get(elements[j])
                limit *= 0.7
                if d < limit:
                    bond_matrix[i, j] = 1
    return bond_matrix


def common_members(list0, list1):
    """Find common elements between two lists.
    
    Parameters
    ----------
    lists0, list1 : list
    Two list to compare.

    Returns
    -------
    common : None or list
        List of common elements.
    """
    set0 = set(list0)
    set1 = set(list1)
    common = list(set0 & set1)
    return common


def find_fragments(elements, geos, unit='bohr'):
    """Find fragments by building a bond matrix
    """
    if len(elements) != len(geos):
        raise ValueError("Number of elements and geos don't match.")
    if unit == 'angstrom':
        geos /= BOHR
    elif unit != 'bohr':
        raise ValueError('`unit` can only be `bohr` or `angstrom`')
    bond_matrix = get_bond_matrix(elements, geos)
    natoms = len(elements)
    frag_list = []
    frags = []
    for iatom in range(natoms):
        previous =  [item for sublist in frag_list for item in sublist]
        nonzeros = np.where(bond_matrix[iatom] == 1)[0]
        if nonzeros.any():
            nonzeros = [iatom] + list(nonzeros)
            common = common_members(previous, nonzeros)
            if not common:
                frag_list.append(nonzeros)
                frags.append(geos[nonzeros])
    return frag_list, frags
