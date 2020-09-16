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
from fdeta.traj_tools import default_charges, find_unique_elements, atom_to_mass
from fdeta.traj_tools import data_from_file, clean_atom_name, atom_to_charge


def get_unique_elements(elements, charges):
    """ Sort charges by atom type.
    It recognizes the different atom types by charges
    and makes the list of list of charges per frame.
    
    Parameters
    ----------
    elements : list
        List of elements per frame.
    charges : dict or list
        Information about charges in trajectory, either
        a dictionary or a list of charges per frame.
    
    """
    luniques = []
    uniques = []
    repeated = {}  # use to count repeated atoms with diff charges
    nframes = len(elements)
    # Case where charges are given in a list per frame
    if isinstance(charges, list):
        if len(elements) != len(charges):
            raise ValueError('Number of frames of `charges` and `elements` do not match')
        for iframe in range(nframes):
            natoms = len(elements[iframe])
            if len(charges[iframe]) != natoms:
                raise ValueError('Number of atoms of `charges` and `elements` do not match')
            for iatom in range(natoms):
                element = elements[iframe][iatom]
                charge = charges[iframe][iatom]
                # check that the element is used in
                if element not in luniques:
                    uelem = UElement(element, charge)
                else:
                    index = luniques.index(element)
                    # Check if charge is different
                    if charge != uniques[index].charge:
                        if element in repeated:
                            repeated[element] +=1
                        else:
                            repeated[element] = 1
                        name = element+str(repeated[element])
                        uelem = UElement(name, charge)
                    else:
                        ulem = uniques[index]
                if uelem.count_frames is None:  # New element
                    uelem.count_frames = [0]*nframes
                    uelem.alloc_traj = []
                    for j in range(nframes):
                        uelem.alloc_traj.append([])
                uelem.count_frames[iframe] += 1
                uelem.total_count += 1
                uelem.alloc_traj[iframe].append(iatom)
                # Finally add it to the list
                if element not in luniques:
                    uniques.append(uelem)
                    luniques.append(element)
                del uelem
    elif isinstance(charges, dict):
        for iframe in range(nframes):
            natoms = len(elements[iframe])
            for iatom in range(natoms):
                element = elements[iframe][iatom]
                charge = charges[element]
                # check that the element is used in
                if element not in luniques:
                    uelem = UElement(element, charge)
                else:
                    index = luniques.index(element)
                    # Check if charge is different
                    if charge != uniques[index].charge:
                        if element in repeated:
                            repeated[element] +=1
                        else:
                            repeated[element] = 1
                        name = element+str(repeated[element])
                        uelem = UElement(name, charge)
                    else:
                        uelem = uniques[index]
                if uelem.count_frames is None:  # New element
                    uelem.count_frames = [0]*nframes
                    uelem.alloc_traj = []
                    for j in range(nframes):
                        uelem.alloc_traj.append([])
                uelem.count_frames[iframe] += 1
                uelem.total_count += 1
                uelem.alloc_traj[iframe].append(iatom)
                # Finally add it to the list
                if element not in luniques:
                    uniques.append(uelem)
                    luniques.append(element)
                del uelem
    else:
        raise TypeError('`charges` must be given as list or dictionary')
    return uniques


class UElement:
    """Unique elements.

    Attributes
    ----------
    name : str
    symbol : str
    charge : float
    zcharge : float
    mass : float
    count_frames : list
    alloc_traj : list(list)
    """
    def __init__(self, name, charge, count_frames=None, alloc_traj=None):
        """ Object with all the information about each element.
        """
        if not isinstance(name, str):
            raise TypeError('`name` must be a string.')
        if not isinstance(charge, float):
            raise TypeError('`charge` must be a float.')
        self.name = name
        self.symbol = clean_atom_name(name)
        self.zcharge = atom_to_charge(self.symbol) 
        self.charge = charge
        self.mass = atom_to_mass(self.symbol)
        self.count_frames = count_frames
        self.alloc_traj = alloc_traj
        self.total_count = 0


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

    def compute_pair_correlation_function(self, grid_range=None, grid_bins=None):
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
        for ielement in self.uniques:
            coords = []
            for iframe in frames:
                for position in ielement.alloc_traj[iframe]:
                    coords.append(self.coordinates[iframe][position]
                coords = np.array(coords, dtype=float)
                assert coords.shape[1] == 3
                # Collecting all coordinates through all frames for a given element ielement
                histogram, hedges = np.histogramdd(coordinates, range=box_range, bins=bins)
                self.pcf[ielement] = histogram
        return self.edges, self.pcf
