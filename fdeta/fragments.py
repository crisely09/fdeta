# -*- coding: utf-8 -*-
#  by CGE, 2020.
"""
Tools for Fragment/Molecule handling.

"""

import os
import numpy as np
import qcelemental as qce
from fdeta.traj_tools import read_xyz_file, read_pqr_file, sort_human
from fdeta.traj_tools import read_xyz_trajectory, read_pqr_trajectory


class Fragment():
    """
    Class to allocate information of a fragment.

    Attributes
    ----------
    atoms : np.ndarray(str)
        Symbols of each atom of the fragment.
    coords : np.ndarray((N,3), dtype=float)
        Coordinates of each atom.
    charges : np.ndarray(N, dtype=float)
        Array with effective charges of each atom.

    Methods
    -------
    from_file(fname)

    """
    def __init__(self, atoms, coords, charges=None):
        """Create a Fragment object.

        Parameters
        ----------
        atoms : list/array(str)
            Symbols of each atom of the fragment.
        coords : list/array(float)
            Coordinates of each atom.
        charges : list/array(float)
            Array with effective charges of each atom.
        """
        # check and set attributes
        if isinstance(atoms, list):
            self.atoms = np.array(atoms, dtype=str)
        elif not isinstance(atoms, np.ndarray):
            raise TypeError('`atoms` should be either a list or a Numpy array of strings')
        else:
            self.atoms = atoms
        if isinstance(coords, list):
            self.coords = np.array(coords, dtype=float)
        elif not isinstance(atoms, np.ndarray):
            raise TypeError('`coords` should be either a list or a Numpy array of float')
        else:
            self.coords = coords
        if charges is not None:
            if isinstance(charges, list):
                self.charges = np.array(charges, dtype=float)
            elif not isinstance(charges, np.ndarray):
                raise TypeError('`charges` should be either a list or a Numpy array of float')
            else:
                self.charges = charges
            if len(self.atoms) != len(self.charges):
                raise ValueError('Wrong number of atoms and charges, they should be the same.')
        else:
            self.charges = None
        if len(self.atoms) != len(self.coords):
            raise ValueError('Wrong number of atoms and coordinates, they should be the same.')
        self.latoms = len(self.atoms)
            

    def __str__(self):
        """To have a nicer way to print them."""
        fstr = ''
        for iatom in range(self.natoms):
            info = [self.atoms[iatom]]
            for axis in range(3):
                info.append(self.coords[iatom, axis])
            if self.charge is not None:
                info.append(self.charges)
                info = tuple(info)
                fstr += '%s\t%.6f\t%.6f\t%.6f\t%.4f\n' % info
            else:
                info = tuple(info)
                fstr += '%s\t%.6f\t%.6f\t%.6f\n' % info
        return fstr

    @classmethod
    def from_file(cls, fname):
        """Read Fragment from file.

        Parameters
        ----------
        fname : str
            Path from where to read the fragment information.
            Note: Reads only files with .xyz or .pqr extensions.

        Returns
        -------
        frag : An instance of fragment class.
        """
        if fname.endswith('.xyz'):
            info = read_xyz_file(fname)
            atoms = info['atoms']
            coords = info['coords']
            charges = None
            return cls(atoms, coords, charges)
        elif fname.endswith('.pqr'):
            info = read_pqr_file(fname)
            atoms = info['atoms']
            coords = info['coords']
            charges = info['charges']
            return cls(atoms, coords, charges)
        else:
            raise NotImplementedError('Only files with xyz or pqr extension are implemented.')


class Ensemble():
    """Fragment information of an ensemble of frames.

    Attributes
    ----------
    fragments : list(Fragment)
        List of fragment objects over the frames.
    
    Methods
    -------
    from_files : Read info from a set of files. 
    """
    def __init__(self, latoms, lcoords, lcharges):
        """Create an ensemble of fragments.

        Parameters
        ----------
        latoms : list
            List of atoms appearing in each frame.
        lcoords : list
            List of arrays with cartesian coordinates for each atom, per frame.
        lcharges: list
            List of the correspongin charges of each atom, in each frame.
        """
        # check all list are the same length
        if not len(latoms) == len(lcoords):
            raise ValueError('The number of `latoms` and `lcoords` should be the same.')
        if lcharges is None:
            lcharges = [None, None]
        if not len(lcharges) == len(lcoords):
            raise ValueError('The number of `lcharges` and `lcoords` should be the same.')
        self.nframes = len(latoms)
        # Make list of fragments
        self.fragments = [Fragment(latoms[iframe], lcoords[iframe], lcharges[iframe]) for iframe in range(self.nframes)]

    @classmethod
    def from_files(cls, folder, basename, extension='pqr'):
        """Generate an ensemble from a set of files.

        Parameters
        ----------
        folder : str
            Path to the files.
        basename : str
            Base name used in all the files to be used.
        extension : str
            Type of file to be loaded. Only `pqr` and `xyz` implemented.
        """
        # Get list of files
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        sort_human(files)
        if extension == 'pqr':
            lfiles = [os.path.join(folder, f) for f in files if f.endswith('.pqr') and basename in f]
            info = read_pqr_trajectory(lfiles)
        if extension == 'xyz':
            lfiles = [os.path.join(folder, f) for f in files if f.endswith('.xyz') and basename in f]
            info = read_xyz_trajectory(lfiles)
            info['charges'] = None
        return cls(info['atoms'], info['coords'], info['charges'])


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


def get_interfragment_distances(frag0, frag1):
    """Get all the distances between the of two fragments

    Parameters
    ----------
    frag0, frag1
        Geometry arrays of each fragment.

    Returns
    -------
    distances : list
        All distances between the two fragments
    """
    distances = []
    if len(frag0.shape) > 1:
        for xyz0 in frag0:
            if len(frag1.shape) > 1:
                for xyz1 in frag1:
                    distances.append(np.linalg.norm(xyz0 - xyz1))
            else:
                distances.append(np.linalg.norm(xyz0 - frag1))
    else:
        if len(frag1.shape) > 1:
            for xyz1 in frag1:
                distances.append(np.linalg.norm(frag0 - xyz1))
        else:
            distances.append(np.linalg.norm(frag0 - frag1))
    return distances
