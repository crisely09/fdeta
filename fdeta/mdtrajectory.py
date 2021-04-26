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
from fdeta.fragment import Fragment, Ensemble
from fdeta.kabsch import centroid, kabsch
from fdeta.uelement import get_unique_elements, UElement
from fdeta.traj_tools import default_charges, find_unique_elements, atom_to_mass
from fdeta.traj_tools import data_from_file, clean_atom_name, atom_to_charge
from fdeta.traj_tools import compute_center_of_mass
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
    # ref_centroid = compute_center_of_mass(masses, reference)
    if not centered:
        cur_centroid = centroid(current)
        # cur_centroid = compute_center_of_mass(masses, current)
        new_current = current - cur_centroid
        new_reference = reference - ref_centroid
        rot_matrix = kabsch(new_current, new_reference)
    else:
        rot_matrix = kabsch(current, reference)
    return ref_centroid, rot_matrix


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
    fragments : Fragment/Ensemble
        Information of each fragment (atoms, coords, charges).
        If instance of Ensemble it contains information
        of more than one frame.
    unique_elements : dict(UElement)
        All the information of unique elements in the trajectory.
    grid_range : np.ndarray([xmin, xmax], [ymin, ymax], [zmin, zmax]).T
        3D Range in each axis.
    grid_bins :  tuple (Nx, Ny, Nz)
        Number of bins used in each axis for histograms.
    """
    def __init__(self, frag0, frag1, aligned=True, grid_range=None, grid_bins=None):
        """

        Parameters
        ----------
        frag0 : Fragment
            Information of fragment 0/A, atoms, coords and charges
        frag1 :  Fragment/Ensemble
            Information of fragment 1/B. Usually a collection of frames as
            Ensemble instance.
        grid_range : np.array(([xmin, xmax], [ymin, ymax], [zmin, zmax])).T 
            Grid needed for making histograms.
        grid_bins : tuple(Nx, Ny, Nz)
            Number (integer) of bins to use on each axis.
        """
        # Standard checks
        if not isinstance(frag0, Fragment):
            raise TypeError('`frag0` must be an instance of Fragment class.')
        if not isinstance(frag1, Ensemble):
            if not isinstance(frag1, Fragment):
                raise TypeError('`frag1` must be either an instance of Fragment or Ensemble classes.')
            else:
                self.nframes = 1
        else:
            self.nframes = frag1.nframes
        self.frag0 = frag0
        self.frag1 = frag1
        # Set grid values for histogram
        self.aligend = aligned
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

    def align_along_trajectory(self, ref_atoms, ref_coords, path_info=None):
        """ Aligns all structures in MD trajectory to the first one.

        Parameters
        ----------

        """
        if path_info is None:
            # Assume it is in the same directory
            path_info = os.getcwd()
        rm_folder = os.path.join(path_info, 'rot_matrices')
        cn_folder = os.path.join(path_info, 'geo_centers')
        if not os.isdir(rm_folder)):
            raise ValueError('Wrong folder path for rot_matrices')
        if not os.isdir(cn_folder):
            raise ValueError('Wrong folder path for geo_centers')
        # Check reference information
        if not (ref_atoms == frag0.atoms).all():
            raise ValueError('Reference atoms must be in the same order as frag0.')
        if not len(ref_coords) == len(frag0.coords):
            raise ValueError('Wrong number of reference atoms/coordinates,  must be the same as frag0.')
        ref_centroid = centroid(ref_coords)
        for iframe in range(self.nframes):
            ref_centroid, rot_matrix = perform_kabsch(ref_coords, frag1.fragments[iframe], centered=False)
            ccent = centroid(frag1.fragments[iframe])
            new_coords = np.dot(new_coords - ccent, rot_matrix)
            ret_coords += ref_centroid


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
        else:
            self.grid_range = grid_range
        if grid_bins is None:
            if self.grid_bins is None:
                raise ValueError('`grid_bins` is missing.')
        else:
            self.grid_bins = grid_bins
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
                    histogram, hedges = np.histogramdd(coords, range=self.grid_range, bins=self.grid_bins)
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
