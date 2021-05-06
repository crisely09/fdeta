#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Created on Sept 2020
#  Trajectory Class
#  @author: C.G.E.
"""
Module for trajectory analysis.

"""

import numpy as np
from fdeta.fragments import Fragment, Ensemble
from fdeta.kabsch import perform_kabsch
from fdeta.uelement import get_unique_elements
from fdeta.traj_tools import find_unique_elements
from fdeta.traj_tools import atom_to_charge
from fdeta.traj_tools import compute_center_of_mass
from fdeta.units import BOHR


def simple_pbc(coords, limits):
    """Apply simple PBC.

    Parameters
    ----------
    coords : np.array(Natoms, 3)
        Cartesian coordinates of all atoms.
    limits : list/array shape(3,2)
        Max and min limits in each cartesian axis.
    """
    result = coords.copy()
    for point in result:
        for axis in range(3):
            if point[axis] < limits[axis, 0]:
                point[axis] += limits[axis, 1]
            elif point[axis] > limits[axis, 1]:
                point[axis] -= limits[axis, 1]
    return result


def get_charge_dist(atoms, geometry, charges, grid_range, grid_bins, zcharge=None,
                     charge_grid=None):
    """ Given the method computes the pair correlation function (histogram)
    of each unique element.

    Parameters
    ---------
    grid_range : np.ndarray(float)
        Range of histogram box
    grid_bins :
        Bin specification for the numpy.histogramdd function. Any of the following:
        1) A sequence of arrays describing the monotonically increasing bin edges
        along each dimension.
        2) The number of bins for each dimension (nx, ny, … =bins)
        3) The number of bins for all dimensions (nx=ny=…=bins).
    zcharge : dict (str: float)
        Saved charges from passed atoms
    """
    if zcharge is None:
        zcharge = {}
        c_new_calc = True
    else:
        if charge_grid is None:
            raise ValueError('Missing `charge_grid` parameter.')
        c_new_calc = False
    for a in atoms:
        if a not in zcharge:
            zcharge[a] = atom_to_charge(a)
    natoms = len(atoms)
    # Find all the atoms, even same element with different charges
    unique_charges = find_unique_elements(charges)
    lucharges = len(unique_charges)
    print('Found %d unique charges: ' % lucharges)
    cstr = '{:.3f}\n'*lucharges
    print(cstr.format(*unique_charges))
    for unique in unique_charges:
        incharges = [iatom for iatom in range(natoms) if unique == charges[iatom]]
        c_hist, c_edges = np.histogramdd(geometry[incharges], range=grid_range, bins=grid_bins)
        if c_new_calc:
            charge_grid = c_hist*unique
            c_new_calc = False
        else:
            charge_grid += c_hist*unique
    return zcharge, charge_grid


def get_charge_and_electron_dist(atoms, geometry, charges, grid_range, grid_bins, zcharge=None,
                                  charge_grid=None, electron_grid=None, log=False):
    """ Given the method computes the pair correlation function (histogram)
    of each unique element.

    Parameters
    ---------
    grid_range : np.ndarray(float)
        Range of histogram box
    grid_bins :
        Bin specification for the numpy.histogramdd function. Any of the following:
        1) A sequence of arrays describing the monotonically increasing bin edges
        along each dimension.
        2) The number of bins for each dimension (nx, ny, … =bins)
        3) The number of bins for all dimensions (nx=ny=…=bins).

    """
    if zcharge is None:
        zcharge = {}
        c_new_calc = True
        a_new_calc = True
    else:
        if charge_grid is None:
            raise ValueError('Missing `charge_grid` parameter.')
        if electron_grid is None:
            raise ValueError('Missing `electron_grid` parameter.')
        c_new_calc = False
        a_new_calc = False
    for a in atoms:
        if a not in zcharge:
            zcharge[a] = atom_to_charge(a)
    natoms = len(atoms)
    # Find all the atoms, even same element with different charges
    unique_charges = find_unique_elements(charges)
    lucharges = len(unique_charges)
    if log:
        print('Found %d unique charge(s): ' % lucharges)
        cstr = '{:.3f}\t'*lucharges
        print(cstr.format(*unique_charges))
    for unique in unique_charges:
        incharges = [iatom for iatom in range(natoms) if unique == charges[iatom]]
        c_hist, c_edges = np.histogramdd(geometry[incharges], range=grid_range, bins=grid_bins)
        if c_new_calc:
            charge_grid = c_hist*unique
            c_new_calc = False
        else:
            charge_grid += c_hist*unique
        # For density  we only need atom type (element)
        anames = [atoms[iatom] for iatom in incharges]
        unique_atoms = find_unique_elements(anames)
        if log:
            print('Found %d unique atom(s): ' % len(unique_atoms))
            cstr = '{:s}\t'*len(unique_atoms)
            print(cstr.format(*unique_atoms))
        for uatom in unique_atoms:
            inatoms = [ind for ind in incharges if atoms[ind] == uatom]
            a_hist, a_edges = np.histogramdd(geometry[inatoms], range=grid_range, bins=grid_bins) 
            if unique > 0:
                ccoeff = zcharge[uatom] - unique
            else:
                ccoeff = zcharge[uatom] + abs(unique)
            if a_new_calc:
                electron_grid = a_hist*ccoeff
                a_new_calc = False
            else:
                electron_grid += a_hist*ccoeff
    return zcharge, charge_grid, electron_grid


def get_density_from_charges(charge_grid, grid_ranges, grid_bins):
    """Evaluate the total and electronic charge from the PCFs.

    Returns
    -------
    charge_density, electron_density : np.array
        Total charge density and electronic density evaluated in the cubic grid used
        for the histogram, already sorted as in the cubic file format.
    """
    # Get the edges as in the np.histogram 
    xv, yv, zv = np.meshgrid(points[1], points[0], points[2])
    delta = np.diff(edges)
    dv = delta[0][0] * delta[1][0] * delta[2][0]

    # Expand in cubic grid-point order
    # and divide by volume element
    charges = contract_grid(charge_grid)
    charge_density = charges / dv
    charge_density *= BOHR**3
    return charge_density
