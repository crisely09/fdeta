#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Created on Sept 2020
#  Trajectory Class
#  @author: C.G.E.
"""
Base Class for trajectory analysis.

"""

import numpy as np
from typing import Union
import fdeta.kabsch as kb
from fdeta.uelement import get_unique_elements, UElement
from fdeta.traj_tools import default_charges, find_unique_elements, atom_to_mass
from fdeta.traj_tools import data_from_file, clean_atom_name, atom_to_charge
from fdeta.units import BOHR


def perform_kabsch(reference, current, centered=False):
    """Get translation matrix and rotation matrix from Kabsch algorithm.
    Finds the optimal rotation matrix to align two geometries that are
    centered on top of each other.

    Parameters
    ----------
    reference : np.array
        Reference geometry to which to align.
    current : np.array
        Current working geometry to be aligned.
    centered : bool
        Whether or the two geometries are already centered
        on top of each other (at the origin).
    """
    # Get translation vector to move to the origin
    ref_centroid = centroid(reference) 
    if not centered:
        cur_centroid = centroid(current)
        reference -= ref_centroid
        current -= cur_centroid
    rot_matrix = kabsch(current, reference)
    return ref_centroid, rot_matrix


def contract_grid(values_hist):
    """Return the expanded values on each grid point.

    Parameters
    ----------
    values_hist
    """
    vshape = values_hist.shape
    nx = vshape[0]
    ny = vshape[1]
    nz = vshape[2]
    result = np.zeros(nx*ny*nz)
    values = values_hist.flatten()
    count = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                fcount = z + y*ny + x*ny*nz
                result[count] = values[fcount]
                count += 1
    return result


def apply_PBC_translation(current, trans_matrix, grid_range):
    """Apply periodic boudary conditions to properly translate molecules.
    """
    raise NotImplementedError


class MDTrajectory:
    """
        Reading a trajectory from XYZ file name, picking a subsystem from XYZ trajectory.

    Attributes
    ----------
    elements : list(list(str))
        List of elements per frame.
    coordinates : list(np.ndarray)
        List of coordinates per frame.
    unique_elements : dict(UElement)
        All the information of unique elements in the trajectory.
    grid_range : np.ndarray([xmin, xmax], [ymin, ymax], [zmin, zmax]).T
        3D Range in each axis.
    grid_bins :  tuple (Nx, Ny, Nz)
        Number of bins used in each axis for histograms.

    """
    def __init__(self, elements, coordinates, charges, grid_range=None, grid_bins=None):
        """

        Parameters
        ----------
        elements : list or list of lists 
            List/array of elements, multiple lists if different 
            for each frame of the trajectory.
        coordinates : 
            Coordinates of each element for each frame.
        charges : dict or list
            Two options: 1) Charges of each unique element, or 2)list of 
            charges per element element for each frame.
        grid_range : np.array(([xmin, xmax], [ymin, ymax], [zmin, zmax])).T 
            Grid needed for making histograms.
        grid_bins : tuple(Nx, Ny, Nz)
            Number (integer) of bins to use on each axis.
        """
        # Standard checks
        if not isinstance(coordinates, list):
            if not isinstance(coordinates, np.ndarray):
                raise TypeError('`coordinates` should be either a list or an numpy array.')
        self.nframes = len(coordinates)
        if not isinstance(elements, list):
            if not isinstance(elements, np.ndarray):
                raise TypeError("`elements` should be either a list or an numpy array.")
        if not isinstance(coordinates[0], np.ndarray):
            coords = []
            addcoords = True
        else:
            addcoords = False
            self.coordinates = coordinates
        if isinstance(elements[0], list):
            if len(elements) != self.nframes:
                raise ValueError('Number of frames of `elements` and `coordinates` do not match')
            self.elements = elements
            if addcoords:
                for i in range(self.nframes):
                    xyztmp = coordinates[i]
                    coords.append(np.array(coordinates[i]))
                    if coords[-1].shape[1] != 3:
                        raise ValueError('Coordinates should be shape (Nelements, 3)')
            self.coordinates = coords
        else:
            els = []
            for i in range(self.nframes):
                els.append(elements)
                if addcoords:
                    xyztmp = coordinates[i]
                    if coordinates.shape[1] != 3:
                        raise ValueError('Coordinates should be shape (Nelements, 3)')
                    coords.append(np.array(coordinates[i]))
            self.elements = els
            self.coordinates = coords
        self.unique_elements = get_unique_elements(self.elements, charges)
        # Set grid values for histogram
        self.grid_range = grid_range
        self.grid_bins = grid_bins
        self.pcf = None

    @staticmethod
    def align(current, trans_matrix, rot_matrix):
        """Align two geometries using matrices from Kabsch algorithm.
        
        Parameters
        ----------
        current : np.ndarray
            Coordinates of molecule to align.
        trans_matrix : np.ndarray
            Translation matrix, usually the centroid of a molecule.
        rot_matrix : np.ndarray
            Rotation matrix result of the Kabsch algorithm.
        """
        # Getting structures to be fitted
        geo = current.copy()
        # Translation of structures
        geo -= trans_matrix
        np.dot(geo, rot_matrix, out=geo)
        return geo

#   def align_along_trajectory(self, frag_id: int, trajectory: dict = None,
#                              to_file: bool = False):
#       """ Aligns all structures in MD trajectory to the first one.

#       Arguments
#       ---------
#       frag_id : int
#           The unique number determining the molecule.
#       trajectory : dictionary
#           trajectory has the same structure as Trajectory.Topology.

#       """
#       if trajectory is None:
#           trajectory = self.trajectory
#       alignment = {}
#       errors = {}
#       geos = {}
#       # Given frag_id, the reference structure is taken from the first frame.
#       reference = self.get_structure_from_trajectory(frag_id, 0, trajectory)
#       for iframe in range(self.nframes):
#           current = self.get_structure_from_trajectory(frag_id, iframe, trajectory)
#           aligned, rmatrix, centroid, rmsd_error = self.align(current, reference)
#           alignment[iframe] = [aligned, rmatrix, centroid]
#           errors[iframe] = rmsd_error
#           geos[iframe] = aligned
#       self.aligned[frag_id] = alignment
#       self.errors[frag_id] = errors
#       if to_file:
#           self.save_trajectory(frag_id, geometries=geos, fname='aligned')
#       else:
#           return alignment

    def compute_pair_correlation_functions(self, grid_range=None, grid_bins=None):
        """ Given the method computes the pair correlation function (histogram)
        of each unique element.

        Parameters
        ---------
        grid_range : np.ndarray(float)
            Range of histogram box
        bins :
            Bin specification for the numpy.histogramdd function. Any of the following:
            1) A sequence of arrays describing the monotonically increasing bin edges
            along each dimension.
            2) The number of bins for each dimension (nx, ny, … =bins)
            3) The number of bins for all dimensions (nx=ny=…=bins).

        """
        self.pcf = {}
        if grid_range is None:
            if self.grid_range is None:
                raise ValueError('`grid_range` is missing.')
        if grid_bins is None:
            if self.grid_bins is None:
                raise ValueError('`grid_bins` is missing.')

        # Make array with coordinates
        for ielement in self.unique_elements:
            fcount = 0
            for iframe in range(self.nframes):
                coords = []
                for position in ielement.alloc_traj[iframe]:
                    coords.append(self.coordinates[iframe][position])
                if coords != []:
                    fcount += 1
                    coords = np.array(coords, dtype=float)
                    assert coords.shape[1] == 3

                    # Collecting all coordinates through all frames for a given element ielement
                    histogram, hedges = np.histogramdd(coords, range=grid_range, bins=grid_bins)
                    if not hasattr(self, 'edges'):
                        self.edges = hedges
                    else:
                        if not np.allclose(hedges, self.edges):
                            raise Warning('There is a discrepancy between previously used grid.')
                    if ielement.name in self.pcf:
                        self.pcf[ielement.name] += histogram
                    else:
                        self.pcf[ielement.name] = histogram
            ielement.frame_count = fcount
        return self.pcf

    def compute_charge_densities(self):
        """Evaluate the total and electronic charge from the PCFs.

        Returns
        -------
        charge_density, electron_density : np.array
            Total charge density and electronic density evaluated in the cubic grid used
            for the histogram, already sorted as in the cubic file format.
        """
        if self.pcf is None:
            self.compute_pair_correlation_functions()
        # Get grid size, assuming all the element PCFs are done in the same grid
        random_element = self.unique_elements[0].name
        electron_grid = np.zeros(self.pcf[random_element].shape)
        charge_grid = electron_grid.copy()
        for uelement in self.unique_elements:
            if uelement.charge > 0:
                ccoeff = uelement.zcharge - uelement.charge
            else:
                ccoeff = uelement.zcharge + abs(uelement.charge)
            electron_grid += ccoeff*self.pcf[uelement.name]/uelement.frame_count
            charge_grid += uelement.charge*self.pcf[uelement.name]/uelement.frame_count
        delta = np.diff(self.edges)
        dv = delta[0][0] * delta[1][0] * delta[2][0]

        # Expand in cubic grid-point order
        # and divide by volume element
        charges = contract_grid(charge_grid)
        electron_density = contract_grid(electron_grid)
        charge_density = charges / dv
        charge_density *= BOHR**3
        electron_density /= dv
        electron_density *= BOHR**3
        return charge_density, electron_density
