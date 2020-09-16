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
from fdeta.traj_tools import default_charges, find_unique_elements
from fdeta.traj_tools import data_from_file


class MDTrajectory:
    """
        Reading a trajectory from XYZ file name, picking a subsystem from XYZ trajectory.

    """
    def __init__(self, elements, coordinates, charges, grid=None, align=False):
        """

        Parameters
        ----------
        elements : 
            List/array of elements for each frame of the trajectory.
            So list of lists or list of arrays.
        coordinates : 
            Coordinates of each element for each frame.
        charges :
            Charges of each element for each frame.
        grid :
            Grid needed for making histograms.
        """

    @classmethod
    def from_file(cls, files: Union[str, List[str]], use_aligned: bool = True):
        return cls(elements, nframes, natoms, unique_ids, uniques, trajectory,
                   eframes, use_aligned=True)

    @classmethod
    def from_data(cls, data: dict, use_aligned: bool = True):
        """
        Read trajectory and trajectory from the files 'name' and 'trajectory'.

        Parameters
        ----------
        data : dict
            Data from trajectory read with one of the tools in traj_tools module.
            Dictionary must contain 'elements', 'geometries', and 'ids' if more than
            one fragment is given.

        Returns
        -------
        MDTrajectory object

        """
        # Save trajectory info
        # data = data_from_file(traj_file)
        if 'ids' in data:
            unique_ids = find_unique_elements(data['ids'])
            unique_ids = [int(u) for u in unique_ids]
        else:
            unique_ids = [0]

        nframes = len(data['elements'])
        natoms = len([item for sublist in data['elements']
                           for item in sublist])
        elements = data['elements']
        uniques = find_unique_elements(data['elements'])
        trajectory = {}
        eframes = {}
        for i in range(nframes):
            elements = np.array(data['elements'][i], dtype=str)
            # Save number of apearances of elements
            for ielement in uniques:
                if ielement in elements:
                    if ielement in eframes:
                        eframes[ielement] += 1
                    else:
                        eframes[ielement] = 1
            geometries = np.array(data['geometries'][i], dtype=float)
            natoms = len(elements)
            if 'ids' in data:
                ids = np.array(data['ids'][i], dtype=int)
                for frag_id in unique_ids:
                    if i == 0:
                        trajectory[frag_id] = dict(geometries=[],
                                                        elements=[])
                    index = np.where(ids == frag_id)[0]
                    trajectory[frag_id]['geometries'].append(geometries[index])
                    trajectory[frag_id]['elements'].append(elements[index])
            else:
                frag_id = 0
                if i == 0:
                    trajectory[frag_id] = dict(geometries=[],
                                                    elements=[])
                trajectory[frag_id]['geometries'].append(geometries)
                trajectory[frag_id]['elements'].append(elements)
        return cls(elements, nframes, natoms, unique_ids, uniques, trajectory,
                   eframes, use_aligned=True)

    def add_fragment(fragment, frag_id=None):
        """ Add fragment to trajectory.

        Parameters
        ----------
        fragment : str or dict
            Information of the new fragment, given
            in a file or as a dictionary of elements and geometries
            per frame. Number of frames needs to match with
            original trajectory.
        frag_id : int
            Fragment ID, to identify it later.
        """
        if frag_id is None:
            frag_id = self.nfrags
            self.nfrags += 1
        else:
            if frag_id in self.frags:
                raise ValueError("""That fragment already exist, for updates use""" 
                                 """ `update_frag_geometries` method.""")
        # If read from file
        if isinstance(fragment, str):
            data = self.data_from_file(fragment)
            self.trajectory[frag_id] = data
            for i in range(self.nframes):
                elements = np.array(data['elements'][i], dtype=str)
                geometries = np.array(data['geometries'][i], dtype=float)
                if i == 0:
                    self.trajectory[frag_id] = dict(geometries=[],
                                                    elements=[])
                self.trajectory[frag_id]['geometries'].append(geometries)
                self.trajectory[frag_id]['elements'].append(elements)
        # If it's already a dictionary
        elif isinstance(fragment, dict):
            if not 'elements' in fragment:
                raise KeyError("Fragment must contain `elements`.")
            if not 'geometries' in fragment:
                raise KeyError("Fragment must contain `geometries`.")
            if len(data['elements']) != len(data['geometries']):
                raise ValueError("Wrong shape or arrange of `elements` and `geometries`")
            nframes = len(data['elements'])
            if nframes != self.nframes:
                raise ValueError("Wrong number of frames of input fragment from dictionary.")
            if isinstance(elements[0], list):
                for i in range(nframes):
                    elements[i] = np.array(elements[i], dtype=str)
                    geometries[i] = np.array(geometries[i], dtype=float)
            self.trajectory[frag_id] = fragment

    def read_charges(self, fname: str):
        """ Read charges from file.
        It saves the charges from a file and takes into account the unique elements.

        Parameters
        ----------
        fname : str
            Name of file which contains the charges, one per element,
            it should match the trajectory.

        """
        # Read charges and find unique elements
        self.charges = {}
        elements = []
        uniques = []
        repeated = {}  # use to count repeated atoms with diff charges
        with open(fname, "r") as fc:
            cdata = fc.readlines()
        for i, line in enumerate(cdata):
            element, charge = line.split()
            # check that the element is used in
            if element not in self.uniques:
                print("""Element not used: %s.""" % element)
            else:
                elements.append(element)
            if element not in self.charges:
                self.charges[element] = float(charge)
                uniques.append(element)
            else:
                if self.charges[element] != float(charge):
                    if element in repeated:
                        repeated[element] += 1
                    else:
                        repeated[element] = 0
                    nelement = element + str(repeated[element])
                    self.charges[nelement] = float(charge)
                    uniques.append(nelement)

    def get_structure_from_trajectory(self, frag_id: int = None,
                                    iframe: int = None,
                                    trajectory: dict = None):
        """
        Given frame number and frag_id returns the ndarray of XYZ coordinates.

        Parameters
        ----------
        frag_id : int
            Molecular ID number
        iframe : int
            Frame number
        trajectory : dict
            Input trajectory

        Returns
        ----------
        trajectory[frag_id][0][iframe] : ndarray

        """

        if None not in (frag_id, iframe, trajectory):
            return trajectory[frag_id]['geometries'][iframe]
        else:
            raise ValueError("Parameters are not specified correctly!")

    def save_trajectory(self, frag_id, frames: list = None,
                      geometries: np.ndarray = None,
                      fname: str = 'trajectory'):
        """ Save the trajectory for a given fragment.

        Parameters
        ----------
        frag_id : int
            ID of fragment to use.
        geometries : dict(np.ndarray((nelements, 3)))
            Geometries of the fragment along trajectory.
            If not given, the initial geometries are used.
        fname : str
            Name for the output file.

        """
        if frames is None:
            frames = range(self.nframes)
        atoms = self.trajectory[frag_id]['elements']
        # Getting Elements from Topology and XYZ from outside
        with open(fname+"_"+str(frag_id)+'.fde', 'w') as fout:
            for iframe in frames:
                natoms = len(atoms[iframe])
                fout.write(str(natoms)+'\n')
                fout.write(str('Frame '+str(iframe)+'\n'))
                if geometries is None:
                    coords = self.get_structure_from_trajectory(frag_id, iframe, self.trajectory)
                else:
                    coords = geometries[iframe]
                for iline in range(natoms):
                    line = ' '.join(coords[iline].astype(str))
                    fout.write(str(atoms[iframe][iline])+' '+line+' '+str(frag_id)+'\n')

    @staticmethod
    def save_snapshot(frag_id: int, iframe: int, elements: np.ndarray,
                      geometry: np.ndarray, basename: str='snapshot'):
        """Save the geometry of a fragment for one single snapshot.

        Parameters
        ----------
        frag_id : int
            Id of the fragment to use.
        iframe : int
            Frame number to save.
        elements : list
            Names of elements in the fragment.
        geometry :  np.ndarray
            Geometry of the fragment in 3d coordinates.
        basename : str
            Base name of file where the snapshot is saved.
        """
        natoms = len(elements)
        with open(basename+"_"+str(frag_id)+'.fde', 'w') as fout:
            fout.write(str(natoms)+'\n')
            fout.write(str('Frame '+str(iframe)+'\n'))
            for iline in range(natoms):
                line = ' '.join(geometry[iline].astype(str))
                fout.write(str(elements[iline])+' '+line+' '+str(frag_id)+'\n')

    @staticmethod
    def align(current: np.ndarray, reference: np.ndarray):
        """Align two geometries using Kabsch algorithm."""
        # Getting structures to be fitted
        geo = current.copy()
        ref_geo = reference.copy()
        # Computation of the centroids
        geo_centroid = kb.centroid(geo)
        ref_centroid = kb.centroid(ref_geo)
        # Translation of structures
        geo -= geo_centroid
        ref_geo -= ref_centroid
        aligned, rmatrix, rmsd_error = kb.kabsch(geo, ref_geo, rmsd_only=False)
        return aligned, rmatrix, geo_centroid, rmsd_error

    def align_along_trajectory(self, frag_id: int, trajectory: dict = None,
                               to_file: bool = False):
        """ Aligns all structures in MD trajectory to the first one.

        Arguments
        ---------
        frag_id : int
            The unique number determining the molecule.
        trajectory : dictionary
            trajectory has the same structure as Trajectory.Topology.

        """
        if trajectory is None:
            trajectory = self.trajectory
        alignment = {}
        errors = {}
        geos = {}
        # Given frag_id, the reference structure is taken from the first frame.
        reference = self.get_structure_from_trajectory(frag_id, 0, trajectory)
        for iframe in range(self.nframes):
            current = self.get_structure_from_trajectory(frag_id, iframe, trajectory)
            aligned, rmatrix, centroid, rmsd_error = self.align(current, reference)
            alignment[iframe] = [aligned, rmatrix, centroid]
            errors[iframe] = rmsd_error
            geos[iframe] = aligned
        self.aligned[frag_id] = alignment
        self.errors[frag_id] = errors
        if to_file:
            self.save_trajectory(frag_id, geometries=geos, fname='aligned')
        else:
            return alignment

    def get_average_structure(self, frag_id: int, method: str = 'zmat', to_file: bool = True):
        """ Given a subsystem,  computes its average structure along the trajectory.

        Arguments
        ---------
        frag_id : int
            The unique number determining the molecule.
        method : str
            Method to use to average. Options are: `coords`, `zmat`.

        """
        if method not in ['coords', 'zmat']:
            raise ValueError("Valid values for `method` are: `coords` or `zmat`.")
        if frag_id in self.aligned:
            align_info = self.aligned[frag_id]
        else:
            align_info = self.align_along_trajectory(frag_id)
        geos = []
        if method == 'coords':
            for iframe in range(self.nframes):
                geos.append(align_info[iframe][0])
            geos = np.array(geos)
            structure_averaged = np.mean(geos, axis=0)
            elements = self.trajectory[frag_id]['elements'][0]
            if to_file:
                self.save_snapshot(frag_id, 0, elements, structure_averaged)
            else:
                return elements, structure_averaged
            

    def compute_pair_correlation_function(self, box_range: tuple, bins: Union[list, np.ndarray],
                                          solute_id=0):
        """ Given the method computes the pair correlation function (pcf)
        for the solvent fragments in 3 steps:
        a) If required get aligned information of solute (centroids and rotation matrices)
        b) Find elements that belong to the solvent
        c) splitting space by bins and measuring the number of a certain type atoms in each
        Should the pcf be computed only for solute molecule

        Parameters
        ---------
        box_range : tuple(float)
            Range of pcf box
        bins : sequence or int
            Bin specification for the numpy.histogramdd function. Any of the following:
            1) A sequence of arrays describing the monotonically increasing bin edges
            along each dimension.
            2) The number of bins for each dimension (nx, ny, … =bins)
            3) The number of bins for all dimensions (nx=ny=…=bins).
        solute_id : int
            The unique number determining the solute molecule. This molecule will be excluded!

        """
        np.set_printoptions(precision=6,  threshold=np.inf)
        self.pcf = {}
        coords = []
        clen = 0
        # If alignment requiered
        if self.use_aligned:
            # Align all structures for solvent fragments
            # First get alignment from solute
            if solute_id in self.aligned:
                align_info = self.aligned[solute_id]
            else:
                align_info = self.align_along_trajectory(solute_id)
            for i, frag_id in enumerate(self.frags):
                if frag_id != solute_id:
                    if "uniques" in self.trajectory[frag_id]:
                        elements = self.trajectory[frag_id]["uniques"]
                    else:
                        elements = self.trajectory[frag_id]["elements"]
                    xyz_traj = self.trajectory[frag_id]["geometries"]
                    for iframe in range(self.nframes):
                        # Translate to centroid
                        xyz = xyz_traj[iframe] - align_info[iframe][2]
                        elen = len(elements[iframe])
                        ielements = elements[iframe].reshape((elen, 1))
                    #   # Rotate geometry
                    #   np.dot(xyz, align_info[iframe][1], out=xyz)
                        coords.append(np.append(ielements, xyz, axis=1))
                        clen += elen
        else:
            # Just take geometries from input
            for i, frag_id in enumerate(self.frags):
                if frag_id != solute_id:
                    if "uniques" in self.trajectory[frag_id]:
                        elements = self.trajectory[frag_id]["uniques"]
                    else:
                        elements = self.trajectory[frag_id]["elements"]
                    xyz_traj = self.trajectory[frag_id]["geometries"]
                    for iframe in range(self.nframes):
                        # Translate to centroid
                        xyz = xyz_traj[iframe]
                        elen = len(elements[iframe])
                        ielements = elements[iframe].reshape((elen, 1))
                        # Rotate geometry
                        coords.append(np.append(ielements, xyz, axis=1))
                        clen += elen

        # Make array with coordinates
        coords = np.array(coords).reshape((clen, 4))
        for ielement in set(self.uniques):
            indices = np.where(coords[:, 0] == ielement)[0]
            # Collecting all coordinates through all frames for a given element ielement
            coordinates = coords[indices, 1:].astype('float64')
            histogram, hedges = np.histogramdd(coordinates, range=box_range, bins=bins)
            # Only saves once the edges because they are the same for all cases.
            # TODO: confirm this statement.
            if self.edges is None:
                self.edges = hedges
            self.pcf[ielement] = histogram
        return self.edges, self.pcf
