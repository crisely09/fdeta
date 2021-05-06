# -*- coding: utf-8 -*-
#  by CGE, 2020.
"""
Tools for Fragment/Molecule handling.

"""

import os
import numpy as np
import qcelemental as qce
from fdeta.kabsch import perform_kabsch, centroid
from fdeta.traj_tools import read_xyz_file, read_pqr_file, sort_human, try_int
from fdeta.traj_tools import read_xyz_trajectory, read_pqr_trajectory


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


def _core_align(rot_matrix, ref_center, work_geo, work_center, return_mat=False):
    """The main operation for aligning.

    Note
    ----
    The aligning is done at the origin, that is why the centers are required.
    Sometimes things are aligned to the geometric center of another molecule/fragment.

    Parameters
    ----------
    ref_geo : np.ndarray((Natoms, 3))
        Cartesian coordinates of the geometry used as reference.
    work_geo : np.ndarray((Natoms, 3))
        Cartesian coordinates of the geometry to be aligned.
    """

    work_geo = np.dot(work_geo - work_center, rot_matrix)
    work_geo += ref_center
    if return_mat:
        return work_geo, rot_matrix
    else:
        return work_geo


def align_one(work_frag, ref_frag, save_matrix=False, mat_path=None,
              save_center=False, frameid=None):
    """Align only one geometry to a reference geometry.

    ref_frag : dict/Fragment
        Information of the reference fragment used for the aligning.
    save_matrices :  bool
        Weather to save the `rot_matrices` and the `centers` to files.
    mat_path : str
        Path where to save the matrices when `save_matrices` is set to True.
    """
    if isinstance(work_frag, Fragment):
        atoms = work_frag.atoms
        work_geo = work_frag.coords
    elif isinstance(work_frag, dict):
        is_ensemble = False
        atoms = work_frag['atoms']
        work_geo = work_frag['coords']
    else:
        raise TypeError("""Geometries to be aligned must be given either as a dict(atoms, geos)"""
                        """ or as an instance of the Ensemble object.""")
    # Check ref_frag types
    if isinstance(ref_frag, Fragment):
        ref_atoms = ref_frag.atoms
        ref_geo = ref_frag.coords
    elif isinstance(ref_frag, dict):
        ref_atoms = ref_frag['atoms']
        ref_geo = ref_frag['coords']
        if isinstance(ref_geo, list):
            tmp = np.array(ref_geo)
            if tmp.shape[1] != 3:
                raise ValueError('Expecting one single geometry, with shape (natoms, 3).')
    ref_center, rot_matrix = perform_kabsch(ref_geo, work_geo, centered=False)
    work_center = centroid(work_geo)
    new_geo = _core_align(rot_matrix, ref_center, work_geo, work_center)
    # Replace values
    if not is_ensemble:
        work_frag['coords'][:] = new_geo.copy()
    else:
        work_frag.coords[:] = new_geo.copy()
    # Save matrices
    if save_matrix:
        if mat_path is None:
            mat_path = os.getcwd()
        rot_path = os.path.join(mat_path, 'rot_matrices')
        if not os.path.isdir(rot_path):
            os.mkdir(rot_path)
        if frameid is None:
            np.savetxt(os.path.join(rot_path, 'rot_matri.txt'), rot_matrix)
        else:
            np.savetxt(os.path.join(rot_path, 'rot_matrix_%d.txt' % frameid), rot_matrix)
        if save_center:
            np.savetxt(os.path.join(mat_path, 'ref_center.txt'), ref_center)


def _align_from_scratch(geos_ensemble, ref_frag, save_matrices=False, mat_path=None):
    """Align a set of geometries comparing to a reference geometry.

    Parameters
    ----------
    geos_ensemble : dict/Ensemble
        Collection of frames with atoms, coordinates (and charges) of a fragment.
    ref_frag : dict/Fragment
        Information of the reference fragment used for the aligning.
    save_matrices :  bool
        Weather to save the `rot_matrices` and the `centers` to files.
    mat_path : str
        Path where to save the matrices when `save_matrices` is set to True.
    """
    # Check geos_ensemble types
    if isinstance(geos_ensemble, Ensemble):
        is_ensemble = True
        latoms = list()
        lcoords = list()
        for iframe in range(geos_ensemble.nframes):
            ifrag = geos_ensemble.fragments[iframe]
            latoms.append(ifrag.atoms)
            lcoords.append(ifrag.coords)
    elif isinstance(geos_ensemble, dict):
        is_ensemble = False
        latoms = geos_ensemble['atoms']
        lcoords = geos_ensemble['coords']
    else:
        raise TypeError("""Geometries to be aligned must be given either as a dict(atoms, geos)"""
                        """ or as an instance of the Ensemble object.""")
    # Check ref_frag types
    if isinstance(ref_frag, Fragment):
        ref_atoms = ref_frag.atoms
        ref_geo = ref_frag.coords
    elif isinstance(ref_frag, dict):
        ref_atoms = np.array(ref_frag['atoms'])
        ref_geo = ref_frag['coords']
        if isinstance(ref_geo, list):
            tmp = np.array(ref_geo)
            if tmp.shape[1] != 3:
                raise ValueError('Expecting one single geometry, with shape (natoms, 3).')
    # Loop over frames
    nframes = len(latoms)
    if nframes != len(lcoords):
        raise ValueError('Number of atoms and coordinates does not match.')
    for iframe in range(nframes):
        # Check with respect to the ref_fragment
        if not (ref_atoms == latoms[iframe]).all():
            raise ValueError('Atoms of frame %d do not correspond to the reference geometry' % iframe)
        work_geo = lcoords[iframe]
        ref_center, rot_matrix = perform_kabsch(ref_geo, work_geo, centered=False)
        work_center = centroid(work_geo)
        new_geo = _core_align(rot_matrix, ref_center, work_geo, work_center)
        # Save matrices
        if save_matrices:
            if mat_path is None:
                mat_path = os.getcwd()
            rot_path = os.path.join(mat_path, 'rot_matrices')
            if not os.path.isdir(rot_path):
                os.mkdir(rot_path)
            if iframe == 0:
                np.savetxt(os.path.join(mat_path, 'ref_center.txt'), ref_center)
            np.savetxt(os.path.join(rot_path, 'rot_matrix_%d.txt' % iframe), rot_matrix)
        # Replace values
        if not is_ensemble:
            geos_ensemble['coords'][iframe][:] = new_geo.copy()
        else:
            geos_ensemble.frag[iframe].coords[:] = new_geo.copy()


def _align_from_matrices(geos_ensemble, mat_path=None):
    """
    Parameters
    ----------
    geos_ensemble : dict/Ensemble
        Collection of frames with atoms, coordinates (and charges) of a fragment.
    mat_path : str
        Path where to save the matrices when `save_matrices` is set to True.
    """
    # Check path
    if mat_path is None:
        mat_path = os.getcwd()
    rot_path = os.path.join(mat_path, 'rot_matrices')
    if not os.path.isdir(rot_path):
        raise ValueError('Missing `rot_matrices` folder')
    # Check geos_ensemble types
    if isinstance(geos_ensemble, Ensemble):
        is_ensemble = True
        latoms = list()
        lcoords = list()
        lframeids = list()
        for iframe in range(geos_ensemble.nframes):
            ifrag = geos_ensemble.fragments[iframe]
            latoms.append(ifrag.atoms)
            lcoords.append(ifrag.coords)
            lframeids.append(ifrag.frameid)
    elif isinstance(geos_ensemble, dict):
        is_ensemble = False
        latoms = geos_ensemble['atoms']
        lcoords = geos_ensemble['coords']
        lframeids = geos_ensemble['frameids']
    else:
        raise TypeError("""Geometries to be aligned must be given either as a dict(atoms, geos)"""
                        """ or as an instance of the Ensemble object.""")
    # Loop over frames
    nframes = len(latoms)
    if nframes != len(lcoords):
        raise ValueError('Number of atoms and coordinates does not match.')
    ref_center = np.loadtxt(os.path.join(mat_path, 'ref_center.txt'))
    for iframe in range(nframes):
        # Read matrices
        frameid = lframeids[iframe]
        work_geo = lcoords[iframe]
        work_center = centroid(work_geo)
        rot_matrix = np.loadtxt(os.path.join(rot_path, 'rot_matrix_%d.txt' % frameid))
        new_geo = _core_align(rot_matrix, ref_center, work_geo, work_center)
        # Replace values
        if not is_ensemble:
            geos_ensemble['coords'][iframe][:] = new_geo.copy()
        else:
            geos_ensemble.fragments[iframe].coords[:] = new_geo.copy()


def align_frames(geos_ensemble, ref_frag=None, mat_path=None, save_matrices=False):
    """Align all the geometries to a reference structure.

    Parameters
    ----------
    geos_ensemble : dict/Ensemble
        Collection of frames with atoms, coordinates (and charges) of a fragment.
    ref_frag : dict/Fragment
        Information of the reference fragment used for the aligning.
    save_matrices :  bool
        Weather to save the `rot_matrices` and the `centers` to files.
    mat_path : str
        Path where to save the matrices when `save_matrices` is set to True.
    """
    # check basics
    if ref_frag is None:
        if mat_path is None:
            raise ValueError("""Please provide either a reference fragment or a path to """
                             """rotation matrices and centers of geometry.""")
        else:
            _align_from_matrices(geos_ensemble, mat_path)
    else:
        _align_from_scratch(geos_ensemble, ref_frag, mat_path=mat_path,
                                   save_matrices=save_matrices)


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
    def __init__(self, atoms, coords, charges=None, frameid=None):
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
        self.frameid = frameid
            

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
        name = fname.split('.')[-1]
        check_number = try_int(name.split('_')[-1])
        if isinstance(check_number, int):
            frameid = check_number
        else:
            frameid = None
        if fname.endswith('.xyz'):
            info = read_xyz_file(fname)
            atoms = info['atoms']
            coords = info['coords']
            charges = None
            return cls(atoms, coords, charges, frameid=frameid)
        elif fname.endswith('.pqr'):
            info = read_pqr_file(fname)
            atoms = info['atoms']
            coords = info['coords']
            charges = info['charges']
            return cls(atoms, coords, charges, frameid=frameid)
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
    def __init__(self, latoms, lcoords, lcharges, lframeids=None):
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
        self.lframeids = lframeids
        if lframeids is None:
            lframeids = [i for i in range(self.nframes)]
        # Make list of fragments
        self.fragments = [Fragment(latoms[iframe], lcoords[iframe], lcharges[iframe],
                                   frameid=lframeids[iframe]) for iframe in range(self.nframes)]

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
            nfrags = len(lfiles)
            info = read_pqr_trajectory(lfiles)
        if extension == 'xyz':
            lfiles = [os.path.join(folder, f) for f in files if f.endswith('.xyz') and basename in f]
            info = read_xyz_trajectory(lfiles)
            info['charges'] = None
        return cls(info['atoms'], info['coords'], info['charges'], lframeids=info['frameids'])
