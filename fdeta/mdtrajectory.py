#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Created on Wed Jan 28 17:35:35 2015
#  Trajectory Class
#  @author: alaktionov
#  Adapted by C.G.E. 2019
"""
Base Class for trajectory analysis.

"""

import numpy as np
from typing import Union
import fdeta.kabsch as kb
from fdeta.traj_tools import default_charges


class MDTrajectory:
    """
        Reading a trajectory from XYZ file name, picking a subsystem from XYZ trajectory.

        Attributes
        ----------
        natoms : int
            Number of atoms in a system.
        frames : ndarray
            3D ndarray, shape(Total_number_of_frames,Number_of_atoms,Coordinate_column).
            To read X,Y,Z,Molecule_ID Coordinate_column must be equal 0,1,2,3 respectively.
        charges : dict
            Type of unique atom (H,C,N,O...) and charge.
        nframes: int
            Total number of frames in trajectory.
        nfrags : int
            Number of fragments present in trajectory.
        topology : dictionary
            {frag_id: {geometries:3D ndarray, elements: ndarray(str), uniques: ndarray(str)}}

    """
    def __init__(self, traj_file: str, charges_file: str = None):
        """
        Read trajectory and topology from the files 'name' and 'topology'.

        Parameters
        ----------
        traj_file :
            Name of the XYZ trajectory file.
            'number of atoms
             number of frame
             element  X Y Z molecule_ID'
        charges_file :
            Name of file with charges for each atom.

        """
        # Read trajectory from file
        with open(traj_file, "r") as f:
            data = f.readlines()

        frags = []
        elements = []
        self.charges = {}
        self.aligned = {}
        self.edges = None
        for i, line in enumerate(data):
            if i == 0:
                self.natoms = np.int(line)
                self.frames = np.zeros((1, self.natoms, 4))
                self.nframes = 1
                current = 0
                atom_count = 0
            elif i == self.nframes*(self.natoms + 2):
                atom_count = 0
                current += 1
                self.nframes += 1
                self.frames = np.append(self.frames, np.zeros((1, self.natoms, 4)), axis=0)
            elif i == current*(self.natoms+2) + 1:
                pass
            else:
                if current == 0:
                    elements.append(line.split()[0])
                self.frames[current][atom_count] = np.float64(line.rsplit()[1:])
                ifrag = self.frames[current][atom_count][-1]
                if ifrag not in frags:
                    frags.append(int(ifrag))
                atom_count += 1
        self.frags = frags
        self.nfrags = len(frags)

        # Save the information in the topology dictionary
        self.elements = elements
        elements = np.array(elements, dtype=str)
        self.topology = {}
        for frag_id in frags:
            index = np.where(self.frames[:, :, 3] == frag_id)  # All indices of subsystem
            natom_frag = len(index[1])//self.nframes
            fragment = np.reshape(self.frames[index],
                                  (self.nframes, natom_frag, 4))
            self.topology[frag_id] = dict(geometries=fragment[:, :, :3],
                                          elements=elements[index[1][:natom_frag]])

        # Save charges of all elements in the system
        if charges_file is None:
            self.charges = default_charges(elements)
        else:
            self.read_charges(charges_file)

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
        elements = []
        uniques = []
        repeated = {}  # use to count repeated atoms with diff charges
        with open(fname, "r") as fc:
            cdata = fc.readlines()
        if len(cdata) != self.natoms:
            assert ValueError("Number of atoms in charge_file does not match with trajectory given.")
        for i, line in enumerate(cdata):
            element, charge = line.split()
            elements.append(element)
            if element not in self.charges:
                self.charges[element] = (i, float(charge))
                uniques.append(element)
            else:
                if self.charges[element][1] != float(charge):
                    if element in repeated:
                        repeated[element] += 1
                    else:
                        repeated[element] = 0
                    nelement = element + str(repeated[element])
                    self.charges[nelement] = (i, float(charge))
                    uniques.append(nelement)
        self.elements = uniques
        uniques = np.array(uniques, dtype=str)
        # Now save topology with the right types of atoms
        for frag_id in self.frags:
            index = np.where(self.frames[:, :, 3] == frag_id)
            self.topology[frag_id]['uniques'] = uniques[index[0]]

    def get_structure_from_topology(self, frag_id: int = None,
                                    iframe: int = None,
                                    topology: dict = None):
        """
        Given frame number and frag_id returns the ndarray of XYZ coordinates.

        Parameters
        ----------
        frag_id : int
            Molecular ID number
        iframe : int
            Frame number
        topology : dict
            Input topology

        Returns
        ----------
        topology[frag_id][0][iframe] : ndarray

        """

        if None not in (frag_id, iframe, topology):
            return topology[frag_id]['geometries'][iframe]
        else:
            print("Parameters are not specified correctly!")

    def save_topology(self, frag_id, frames: list = None,
                      geometries: np.ndarray = None,
                      fname: str = 'topology_'):
        """ Save the topology for a given fragment.

        Parameters
        ----------
        frag_id : int
            ID of fragment to use.
        geometries : dict(np.ndarray((nelements, 3)))
            Geometries of the fragment along topology.
            If not given, the initial geometries are used.
        fname : str
            Name for the output file.

        """
        if frames is None:
            frames = range(self.nframes)
        atoms = self.topology[frag_id]['elements']
        natoms = len(atoms)
        # Getting Elements from Topology and XYZ from outside
        with open(fname+str(frag_id)+'.txt', 'w') as fout:
            for iframe in frames:
                fout.write(str(natoms)+'\n')
                fout.write(str('Frame '+str(iframe)+'\n'))
                if geometries is None:
                    coords = self.get_structure_from_topology(frag_id, iframe, self.topology)
                else:
                    coords = geometries[iframe]
                for iline in range(natoms):
                    line = ' '.join(coords[iline].astype(str))
                    fout.write(str(atoms[iline])+' '+line+' '+str(frag_id)+'\n')

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
        aligned, rmsd_error = kb.kabsch(geo, ref_geo, output=True)
        return aligned, rmsd_error

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
            trajectory = self.topology
        alignment = {}
        errors = {}
        # Given frag_id, the reference structure is taken from the first frame.
        reference = self.get_structure_from_topology(frag_id, 0, trajectory)
        for iframe in range(self.nframes):
            current = self.get_structure_from_topology(frag_id, iframe, trajectory)
            aligned, rmsd_error = self.align(current, reference)
            alignment[iframe] = aligned
            errors[iframe] = rmsd_error
        self.aligned[frag_id] = alignment
        if to_file:
            self.save_topology(frag_id, geometries=alignment, fname='aligned_')
        else:
            return alignment, errors

    def get_average_structure(self, frag_id: int, method: str = 'zmat'):
        """ Given a subsystem,  computes its average structure along the trajectory.

        Arguments
        ---------
        frag_id : int
            The unique number determining the molecule.
        method : str
            Method to use to average. Options are: `coords`, `zmat`.

        """
        if method == 'coordinates':
            for iframe in range(self.Total_number_of_frames):
                self.structure_averaged = np.mean(np.array(list(self.alignement.values()))[:, 0], axis=0)
            self.save(frag_id, self.structure_averaged)

    def compute_pair_correlation_function(self, box_range: tuple, bins: Union[list, np.ndarray],
                                          solute_id=0):
        """ Given the method computes the pair correlation function (pcf)
        for the solvent fragments in 3 steps:
        a) Get aligned geometries
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
        # Get all aligned structures for solvent fragments
        for i, frag_id in enumerate(self.frags):
            if frag_id != solute_id:
                # Check if it was aligned before
                if frag_id in self.aligned:
                    aligned = self.aligned[frag_id]
                else:
                    aligned = self.align_along_trajectory(frag_id)[0]
                if "uniques" in self.topology[frag_id]:
                    elements = self.topology[frag_id]["uniques"]
                else:
                    elements = self.topology[frag_id]["elements"]
                elen = len(elements)
                elements = elements.reshape((elen, 1))
                for iframe in range(self.nframes):
                    coords.append(np.append(elements, aligned[iframe], axis=1))
                    clen += elen
        # Make array with coordinates
        coords = np.array(coords).reshape((clen, 4))
        for ielement in set(self.elements):
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
