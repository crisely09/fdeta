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
import re


class Trajectory:
    """
        Reading a trajectory from XYZ file name, picking a subsystem from XYZ trajectory.

        Attributes
        ----------
        data : str
            A name of the XYZ trajectory file.
            Trajectory File Format: 'Element  X Y Z Molecule_ID'
        Number_of_atoms : int
            Number of atoms in a system.
        Frames : ndarray
            3D ndarray, shape(Total_number_of_frames,Number_of_atoms,Coordinate_column).
            To read X,Y,Z,Molecule_ID Coordinate_column must be equal 0,1,2,3 respectively.
        Elements : numpy string array
            Type of atom (H,C,N,O...).
        Current_frame : int
            Counter of the current frame number.
        Total_number_of_frames: int
            Total number of frames in trajectory.
        Atom_counter : int
            Counter of the current atom number.
        Size_of_cell : ndarray
            1D array [sizeX,sizeY,sizeZ]. Size of the cell used in MD simulation in Angstrom.
        Subsystems : list
            List of subsystems picked up from Trajectory
        Topology : dictionary
            {List of Molecular_ID:(3D ndarray, tuple(shape(Total_number_of_frames,
            Number_of_selected_atoms,Coordinate_column),Elements,Total_number_of_selected_atoms)}
        index : ndarray
            Numpy array of indices of array Frames where 'topo'==Molecular_ID.
        Number_of_selected_atoms : int
            Number of atoms in subsystem.
        Subsystem : ndarray
            3D ndarray, shape(Total_number_of_frames,Number_of_selected_atoms,Coordinate_column).
        Index_of_elements : dictionary
            Dictionary associates the element's name ('H', 'C'...) with the position
            of the element in coordinate array (e.g. Frame[iframe])


    """

    def __init__(self, name):
        """
        Read trajectory and topology from the files 'name' and 'topology'.

        Parameters
        ----------
        name : str
            A name of the XYZ trajectory file.
            Trajectory File Format: 'Element  X Y Z Molecule_ID'

        """
        try:
            with open(name, "r") as f:
                self.data = f.readlines()
        except IOError as e:
            print("IOError({0}): {1}".format(e.errno, e.strerror))

        for i, line in enumerate(self.data):
            if i == 0:
                self.Number_of_atoms = np.int(line)
                self.Frames = np.zeros((1, self.Number_of_atoms, 4))
                self.Elements = np.asarray([x.rsplit()[0] for x in self.data[2:self.Number_of_atoms + 2]])
                self.Current_frame = 0
                self.Total_number_of_frames = 1
                self.Atom_counter = 0
            elif i == 1:
                self.Size_of_cell = np.float64(re.findall(r"[-+]?\d*\.\d+|\d+", line))
            elif i == self.Total_number_of_frames*(self.Number_of_atoms + 2):
                self.Atom_counter = 0
                self.Current_frame += 1
                self.Total_number_of_frames += 1
                self.Frames = np.append(self.Frames, np.zeros((1, self.Number_of_atoms, 4)), axis=0)
            elif i == self.Current_frame*(self.Number_of_atoms+2) + 1:
                pass
            else:
                # Check what happens if there is no molecular marker in trajectory file
                self.Frames[self.Current_frame][self.Atom_counter] = np.float64(line.rsplit()[1:])
                self.Atom_counter += 1
        self.Topology = {}
        self.Index_of_elements = {}

    def select(self, topo):
        """
        Given a Molecular_ID, 'select' picks up the XYZ coordinates and appends
        to the 'Subsystems' list.

        Parameters
        ---------
        topo : int
            Molecular ID

        """

        self.index = np.where(self.Frames[:, :, 3] == topo)  # All indices of subsystem
        self.Number_of_selected_atoms = len(self.index[1])//self.Total_number_of_frames
        self.Subsystem = np.reshape(self.Frames[self.index],
                                    (self.Total_number_of_frames,
                                     self.Number_of_selected_atoms, 4))
        self.Topology[topo] = self.Subsystem.copy(), self.Elements[self.index[1][:self.Number_of_selected_atoms]], self.Number_of_selected_atoms

    def save(self, *structure):
        """
        Save each topology structure into the XYZ trajectory file named "topo"+Molecular_ID.

        Parameters
        ----------
        structure : list of topologies

        Attributes
        ----------
        k : int
            Counter over the Molecular_ID
        iframe : frame number
            v : touple
            (3D ndarray of cordinates, 1D array of Elements)
        iline : int
            Line number in a given frame iframe
        line : str
            String line containing a name of element in frame 'iframe' on line 'iline'

        """

        if len(structure) == 0:
            # Getting data from topology
            for k, v in self.Topology.iteritems():
                with open('topo'+str(k), 'w') as f:
                    for iframe in range(self.Total_number_of_frames):
                        f.write(str(v[2])+'\n')
                        f.write(str('Frame '+str(iframe)+'\n'))
                        for iline in range(v[2]):
                            line = ' '.join(v[0][iframe][iline].astype(str))
                            f.write(str(v[1][iline])+' '+line+'\n')

        if len(structure) == 2:
            # Getting Elements from Topology and XYZ from outside
            m_id, coordinates = structure[0], structure[1]
            number_of_atoms = self.Topology[m_id][2]
            elements = self.Topology[m_id][1].reshape((1, number_of_atoms))
            np.savetxt('average'+str(m_id), np.concatenate((elements.T, coordinates), axis=1),
                       header='Averaged structure '+str(m_id), comments=str(number_of_atoms)+'\n', fmt="%s")

        if (len(structure) == 3) and structure[2] == 'ALIGNED':
            # Saving trajectory aligned withto the first frame
            m_id, data, key = structure
            number_of_atoms = len(data[0][0])
            elements = self.Topology[m_id][1]
            with open('aligned'+str(m_id), 'w') as f:
                for iframe, coordinates in data.items():
                    f.write(str(number_of_atoms)+'\n')
                    f.write('Frame '+str(iframe)+'\n')
                    for name, iline in zip(elements, coordinates[0]):
                        line = name+' '+' '.join(iline.astype(str))+'\n'
                        f.write(line)

    def get_structure_from_topology(self, molecular_id=None, iframe=None, topology=None):
        """
        Given frame number and molecular_ID returns the ndarray of XYZ coordinates.

        Parameters
        ----------
        molecular_ID : int
            Molecular ID number
        iframe : int
            Frame number
        topology : dict
            Input topology

        Returns
        ----------
        topology[molecular_id][0][iframe] : ndarray

        """

        if None not in (molecular_id, iframe, topology):
            return topology[molecular_id][0][iframe]
        else:
            print("Parameters are not specified correctly!")
