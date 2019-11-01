# -*- coding: utf-8 -*-
#  Created on Thu Feb  5 16:04:28 2015
#  @author: alaktionov
"""
Trajectory Analysis Class.

"""

import numpy as np
import fdeta.kabsch as kb
from fdeta.base_trajectory import Trajectory


class TrajectoryAnalysis(Trajectory):
    """Analysis of Trajectories."""

    def compute_center_of_mass(self, mass, coordinates):
        return np.sum(np.multiply(coordinates, mass.reshape(len(mass), 1)), axis=0)/mass.sum()

    def nametomass(self, a):
        if a == 'H':
            return 1.00794
        elif a == 'O':
            return 15.9994
        elif a == 'C':
            return 12.0107
        elif a == 'N':
            return 14.0067
        elif a == 'S':
            return 32.065

    def nametocharge(self, a):
        if a == 'H':
            return 1.0
        elif a == 'O':
            return 8.0
        elif a == 'C':
            return 6.0
        elif a == 'N':
            return 7.0
        elif a == 'S':
            return 16.0

    def rmsd(self, x, y):
        """Computes RMSD between two set of coordinates X and Y

        Arguments
        ----------
        x : ndarray
            A 2D matrix of XYZ coordinates [X Y Z], shape(Number_of_atoms, 3)
        y : ndarray
            A 2D matrix of XYZ coordinates [X Y Z], shape(Number_of_atoms, 3)

        """

        return np.linalg.norm(x-y)/(len(x)**0.5)

    def align(self, structure, reference_structure):
        """The Kabsch algorithm

        http://en.wikipedia.org/wiki/Kabsch_algorithm

        The algorithm starts with two sets of paired points of structure P and
        reference_structure Q. It returns a rotation matrix U prividing the
        best fit between the two structures,  a measure of that fit RMSD and a
        new geometry of structure P aligned to reference_structure Q.
        Each vector set P and Q is represented as an NxD matrix,  where D is the
        dimension of the space and N is the number of atoms

        The algorithm works in three steps:
        - a translation of P and Q
        - the computation of a covariance matrix C
        - computation of the optimal rotation matrix U

        The optimal rotation matrix U is then used to
        rotate P unto Q so the RMSD can be caculated
        from a straight forward fashion.

        Arguments
        ----------
        structure : ndarray
            2D ndarray [Number_of_atoms,  XYZ coordinates] contains a geometry of structure.
        reference_structure : ndarray
            2D ndarray [Number_of_atoms,  XYZ coordinates] contains a geometry of reference_structure.

        Returns
        ----------
        structure_aligned : ndarray
            2D ndarray shape(Number_of_atoms,  XYZ coordinates) contains a transformed
            coordinates of 'structure' aligned to 'reference_structure'
        U : ndarray
            3D ndarray the ebst rotation matrix.
        rmsd : float64
            RMSD between 'structure_aligned' and 'reference_structure'.
        structure_centroid : ndarray
            1D numpy array containing the coordinates of the 'structure' centroid.
        reference_structure_centroid : ndarray
            1D array containing the coordinates of the 'reference_structure' centroid.

        """

        # Getting structures to be fitted
        self.structure = structure.copy()
        self.reference_structure = reference_structure.copy()
        # Computation of the centroids
        self.structure_centroid = kb.centroid(self.structure)
        self.reference_structure_centroid = kb.centroid(self.reference_structure)
        # Translation of structures
        self.structure -= self.structure_centroid
        self.reference_structure -= self.reference_structure_centroid
        # Get initial RMSD
        # Computation of the covariance matrix
        self.covariation_matrix = np.dot(np.transpose(self.structure), self.reference_structure)

        # Computation of the optimal rotation matrix
        # This can be done using singular value decomposition (SVD)
        # Getting the sign of the det(V)*(W) to decide
        # whether we need to correct our rotation matrix to ensure a
        # right-handed coordinate system.
        # And finally calculating the optimal rotation matrix U
        # see http://en.wikipedia.org/wiki/Kabsch_algorithm
        V,  S,  W = np.linalg.svd(self.covariation_matrix)
        if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        # Create Rotation matrix U
        U = np.dot(V,  W)

        # Rotate P
        self.structure_aligned = np.dot(self.structure,  U)
        return np.asarray([self.structure_aligned, U,
                           self.rmsd(self.structure_aligned, self.reference_structure),
                           self.structure_centroid, self.reference_structure_centroid])

    def align_geometries_from_files(self, structure, reference_structure):
        self.structure_atoms, self.structure = kb.get_coordinates(structure)
        self.reference_structure_atoms, self.reference_structure = kb.get_coordinates(reference_structure)

        self.structure_centroid = kb.centroid(self.structure)
        self.reference_structure_centroid = kb.centroid(self.reference_structure)

        self.structure -= self.structure_centroid
        self.reference_structure -= self.reference_structure_centroid

        self.structure_aligned, self.rmsd_error = kb.kabsch(self.structure, self.reference_structure, output=True)
        self.structure_aligned += self.reference_structure_centroid

    def align_along_trajectory(self, molecular_id, trajectory):
        """ Aligns all structures in MD trajectory to the first one.

        Arguments
        ---------
        molecular_id : int
            The unique number determining the molecule.
        trajectory : dictionary
            trajectory has the same structure as Trajectory.Topology.

        """

        self.alignement = {}
        # Given molecular_id, the reference structure is taken from the first frame.
        Reference_structure = self.get_structure_from_topology(molecular_id, 0, trajectory)[:, :3]
        for iframe in range(self.Total_number_of_frames):
            Current_structure = self.get_structure_from_topology(molecular_id, iframe, trajectory)[:, :3]
            self.alignement[iframe] = self.align(Current_structure, Reference_structure)
        self.save(molecular_id, self.alignement, 'ALIGNED')

    def get_average_structure(self, molecular_id):
        """ Given a subsystem,  computes its average structure along the trajectory.

        Arguments
        ---------
        molecular_id : int
            The unique number determining the molecule.

        Variables
        ---------
        iframe : int
            frame counter

        """

        for iframe in range(self.Total_number_of_frames):
            self.structure_averaged = np.mean(np.array(list(self.alignement.values()))[:, 0], axis=0)
        self.save(molecular_id, self.structure_averaged)

    def compute_pair_correlation_function(self, pcf_range, bins, solute_id=0):
        """ Given the 'alignement' variable the method computes the pair correlation function (pcf)
        for the entire system in 3 steps:
        a) translation to the solute centroid
        b) rotation
        c) splitting space by bins and measuring the number of a certain type atoms in each
        Should the pcf be computed only for solute molecule

        Parameters
        ---------
        pcf_range : tuple(float)
            Range of pcf box
        bins : sequence or int
            Bin specification for the numpy.histogramdd function. Any of the following:
            1) A sequence of arrays describing the monotonically increasing bin edges
            along each dimension.
            2) The number of bins for each dimension (nx, ny, … =bins)
            3) The number of bins for all dimensions (nx=ny=…=bins).
        solute_id : int
            The unique number determining the molecule. This molecule will be excluded!

        """
        np.set_printoptions(precision=6,  threshold=np.inf)
        self.pcf = {}
        # Translation to the geometrical center of rotation and rotation
        tmpframes = []
        for iframe in range(self.Total_number_of_frames):
            tmpframes.append(np.dot(self.Frames[iframe][:, :3]-self.alignement[iframe][3],
                                    self.alignement[iframe][1]))
        self.Frames_aligned = np.asarray(tmpframes)
        del tmpframes
        # Getting indicies for solute (molecules to be excluded)
        self.Solute_index = np.where(self.Frames[0, :, 3] == solute_id)[0]
        # Getting indicies for each type of element
        self.edges = None
        for ielement in set(self.Elements):
            self.Index_of_elements[ielement] = np.where(self.Elements == ielement)[0]
            # Excluding atoms of solute from the list of atoms
            mask = np.in1d(self.Index_of_elements[ielement], self.Solute_index, invert=True)
            self.Index_of_elements[ielement] = self.Index_of_elements[ielement][mask]
            # Collecting all coordinates through all frames for a given element ielement
            fsize = self.Total_number_of_frames*self.Index_of_elements[ielement].size
            coordinates = self.Frames_aligned[:, self.Index_of_elements[ielement]].reshape(fsize, 3)
            histogram, hedges = np.histogramdd(coordinates, range=pcf_range, bins=bins)
            # Only saves once the edges because they are the same for all cases.
            # TODO: confirm this statement.
            if not self.edges:
                self.edges = hedges
            self.pcf[ielement] = histogram
        return self.edges, self.pcf
