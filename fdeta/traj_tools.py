# -*- coding: utf-8 -*-
#  by CGE, 2020.
"""
Tools for MDTrajectory Class.

"""

import re
import os
import numpy as np
from typing import Union
from qcelemental import periodictable



def try_int(s):
    """Check if a variable is integer.

    Parameters
    ----------
    s : var
        The variable to check.
    """
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [try_int(c) for c in re.split('([0-9]+)', s) ]


def sort_human(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def compute_center_of_mass(mass: np.ndarray, coordinates: np.ndarray) -> float:
    return np.sum(np.multiply(coordinates, mass.reshape(len(mass), 1)), axis=0)/mass.sum()


def atom_to_mass(atom: Union[str, int]) -> float:
    """Give back the mass of a given element."""
    return periodictable.to_mass(atom)


def atom_to_charge(atom: Union[str, int]) -> float:
    """Give the nucleus atomic charge."""
    return float(periodictable.to_atomic_number(atom))


def default_charges(elements: list) -> dict:
    """Get the default charges for a list of given elements."""
    charges = {}
    for i, atom in enumerate(elements):
        charges[atom] = atom_to_charge(atom)
    return charges


def flatten_list(chain_list: list) -> list:
    """Flatten recursively a list."""
    is_1d = all(not isinstance(x, list) for x in chain_list)
    if is_1d:
        return chain_list
    flat_list = []
    for sublist in chain_list:
        if isinstance(sublist, list):
            flat_list += flatten_list(sublist)
        else:
            flat_list += [sublist]
    return flat_list


def find_unique_elements(elements: list) -> list:
    """Find unique elements of a list."""
    flat_list = flatten_list(elements)
    uniques = list(set(flat_list))
    return uniques


def clean_atom_name(element: str,
                    heteroatoms: Union[list, np.ndarray] = None) -> str:
    """
    Parameters
    ----------
    element: str
        the element specification from the .pqr
    heteroatoms: list
        The possible heteroatoms present in the system

    Returns
    -------
    str
        the element symbol only
    """
    if heteroatoms is None:
        heteroatoms = ['He', 'Na', 'Mg', 'Si', 'Cl', 'Ca', 'Br']
    if len(element) > 1:
        if element[:2] in heteroatoms:
            return element[:2]
        else:
            return element[0]
    else:
        return element


def get_data_lines(files: Union[str, list]):
    """Get lines of file or files.

    Parameters
    ---------- 
    files : str or list
        File name(s) from where to read the data.

    Returns
    -------
    data : list
        List with all lines.
    """
    if not isinstance(files, list):
        # One file only
        if not isinstance(files, str):
            raise TypeError('Only list or str are valid.')
        with open(files, 'r') as finp:
            data = finp.readlines()
    elif isinstance(files, (list, str)):
        data = []
        for fname in files:
            if not isinstance(fname, str):
                raise TypeError('Only list or str are valid.')
            with open(fname, 'r') as finp:
                lines = finp.readlines()
                data += lines
    else:
        raise TypeError("`files` must be either list or str")
    return data


def read_xyz_file(fname: str) -> dict:
    """Read info from one.

    Parameters
    ----------
    fname :  str
       Path to file to read.

    Returns
    -------
    geos : dict
        Elements, geometries and ids lists.
    """
    data = get_data_lines(fname)
    atoms = []
    coords = []
    atomic_info = 4
    # Separate information
    for i, line in enumerate(data):
        splits = line.split()
        if i == 0:
            natoms = int(line.strip())
        elif i == 1:
            pass
        else:
            if not len(splits) == atomic_info:
                raise ValueError("""This is not a standard -one molecule- file."""
                                 """ See read_xyz_trajectory for multiple molecules.""")
            atoms.append(splits[0])
            xyz = [float(x) for x in splits[1:4]]
            coords.append(xyz)
    if natoms != len(atoms):
        raise ValueError('Wrong number of atoms in file.')
    if natoms != len(coords):
        raise ValueError('Wrong number of coordinates of atoms in file.')
    atoms = np.array(atoms)
    coords = np.array(coords)
    geos = dict(atoms=atoms, coords=coords)
    return geos


def read_xyz_trajectory(files: Union[str, list],
                        has_ids: bool = False) -> dict:
    """Read info from one or multiple files.

    Parameters
    ----------
    files :  str or list(str)
        File or list of files to read.

    Returns
    -------
    geos : dict
        Elements, geometries and ids lists.
    """
    data = get_data_lines(files)
    atoms = []
    coords = []
    if has_ids:
        ids = []
        atomic_info = 5
    else:
        atomic_info = 4
    # Separate information
    for i, line in enumerate(data):
        splits = line.split()
        if len(splits) == 1:
            new_geo = False
            atoms.append([])
            coords.append([])
            if has_ids:
                ids.append([])
        elif len(splits) == atomic_info:
            atoms[-1].append(splits[0])
            xyz = [float(x) for x in splits[1:4]]
            coords[-1].append(xyz)
            if has_ids:
                ids[-1].append(splits[4])
        else:
            pass
    if has_ids:
        geos = dict(atoms=atoms, coords=coords, ids=ids)
    else:
        geos = dict(atoms=atoms, coords=coords)
    return geos


def read_pqr_file(fname: str) -> dict:
    """Read info from one or multiple files.

    Parameters
    ----------
    files :  str or list(str)
        File or list of files to read.
    """
    data = get_data_lines(fname)
    atoms = []
    coords = []
    charges = []
    for n, line in enumerate(data):
        splits = line.split()
        if "ATOM" in splits[0]:
            atoms.append(splits[2])
            coords.append([float(x) for x in splits[5:8]])
            charges.append(float(splits[8]))
        else:
            pass
    # Clean last element of list
    info = dict(atoms=atoms, coords=coords, charges=charges)
    return info


def read_pqr_trajectory(files: Union[str, list]) -> dict:
    """Read info from one or multiple files.

    Parameters
    ----------
    files :  str or list(str)
        File or list of files to read.
    """
    data = get_data_lines(files)
    atoms = [[]]
    coords = [[]]
    charges = [[]]
    for n, line in enumerate(data):
        splits = line.split()
        if len(splits) == 1:
            atoms.append([])
            coords.append([])
            charges.append([])
        elif "ATOM" in splits[0]:
            atoms[-1].append(splits[2])
            coords[-1].append([float(x) for x in splits[5:8]])
            charges[-1].append(float(splits[8]))
        else:
            pass
    # Clean last element of list
    atoms.pop()
    coords.pop()
    charges.pop()
    info = dict(atoms=atoms, coords=coords, charges=charges)
    return info


def read_gromacs_trajectory(files: Union[str, list],
                            solute: Union[list, np.ndarray]) -> dict:
    """Read info from one or multiple files.

    Parameters
    ----------
    files :  str or list(str)
        File or list of files to read.
    solute : list or np.ndarray
        Indices of the solute fragment

    Returns
    -------
    data : dict
        All data of trajectory in `elements`, `geometries`
        and `ids`.

    """
    import MDAnalysis
    files = tuple(files)
    u = MDAnalysis.Universe(*files)
    elements = u.atoms.names
    natoms = elements.size
    elements = [clean_atom_name(e) for e in elements]
    nframes = len(u.trajectory)
    geometries = []
    elements_all = []
    ids = []
    id_tmp = [0 if i in solute else 1 for i in range(natoms)]
    for nf in range(nframes):
        xyz = u.trajectory[nf].positions.copy()
        geometries.append(xyz)
        elements_all.append(elements)
        ids.append(id_tmp)
        
    # Define dictionary
    data = dict(elements=elements_all, geometries=geometries, ids=ids)
    return data


def data_from_file(filename:str) -> dict:
    """Read data from trajectory file.

    Parameters
    ----------
    filename : str
        Name of trajectory file.

    Returns
    -------
    data : dict
        All data of trajectory in `elements`, `geometries`
        and if `.fde` format `ids`.

    """
    # Read trajectory from file
    if filename.endswith('.fde'):
        data = read_xyz_trajectory(filename, has_ids=True)
    else:
        if filename.endswith('.xyz'):
            data = read_xyz_trajectory(filename)
        elif filename.endswith('.pqr'):
            data = read_pqr_trajectory(filename)
        else:
            raise ValueError("""Extension of filename not valid."""
                             """ Try `.xyz`, `.pqr` or `.fde`""")
    return data


def make_trajectory_file(file_root: str, input_format='xyz',
                         output: str = 'trajectory.xyz'):
    """Combine a set of files into ONE xyz file.

    Parameters
    ----------
    file_root: str
        Root of the files to use, relative or ... path.
    input_format : str
        Type of files to be read, options are: `pqr`, `xyz`,
        GROMACS `gro` (for this option another file `.trr` or `.xtc`
        with the same name root must be present in the folder).
    """
    # Get all files with same root and correct extension
    fs = [f for f in os.listdir() if f.startswith(file_root)]
    if input_format == 'gro':
        import MDAnalysis
        inputs = ['.gro', '.trr', '.xtc']
        files = [f for f in fs if f.endswith(inp) for inp in inputs]
        geos = read_gromacs_trajectory(files)
    else:
        files = [f for f in fs if f.endswith(input_format)]
        # Sort files appropriately
        files.sort()
        # Read files
        if input_format == 'xyz':
            geos = read_xyz_trajectory(files)
        elif input_format == 'pqr':
            geos = read_pqr_trajectory(files)
        else:
            raise NotImplementedError("Files with `%s` extension are not supported."
                                      % input_format)
    elements = geos['elements']
    geometries = geos['geometries']
    with open(output, 'w') as ofile:
        nframes = len(elements)
        for n in range(nframes):
            natoms = len(elements[n])
            ofile.write("%d\n" % natoms)
            ofile.write("NFrame = %d\n" % n)
            for iatom in range(natoms):
                values = (elements[n][iatom], geometries[n][iatom][0],
                          geometries[n][iatom][1], geometries[n][iatom][2])
                ofile.write("%s\t%.8f\t%.8f\t%.8f\n" % values)


def write_xyz_file(elements, geometry, fname='molecule.xyz'):
    """Write xyz file from a geometry.

    Parameters
    ----------
    elements : list/array
        List of atomic symbols
    geometry :  np.ndarray
        Cartesian coordinates of the atoms in the same order as elements.
    fname : str
        Name of output file.
    """
    natoms = len(elements)
    with open(fname, 'w') as ofile:
        ofile.write("%d\n" % natoms)
        ofile.write("NFrame = 0\n")
        for iatom in range(natoms):
            values = (elements[iatom], geometry[iatom, 0],
                      geometry[iatom, 1], geometry[iatom, 2])
            ofile.write("%s\t%.8f\t%.8f\t%.8f\n" % values)


def check_length_trajectories(data: list) -> int:
    """Compare the number of frames of two sets of geometries.
    
    Parameters
    ----------
    data : list(dict)
        Geometries of a different trajectories, read from files,
        containing the keys: `elements`, `geometries`.

    Returns
    -------
    nframes : int
        If the trajectories are the same length, return the number
        of frames.
        Else: raise ValueError
    """
    nframes = len(data[0]["elements"])
    for geo in data[1:]:
        if len(geo["elements"]) != nframes:
            raise ValueError("Trajectories don't match number of frames.")
    return nframes


def combine_fragment_files(files: list, output :str = 'all_fragments.fde'):
    """Combine fragments from separate files.

    Parameters:
    -----------
    files : list(str)
        Name/path of files to combine.
    """
    if not isinstance(files, list):
        raise TypeError("Filenames must be given in a list.")
    allgeos = [read_xyz_trajectory(f) for f in files]
    nframes = check_length_trajectories(allgeos)
    print(nframes)
    with open(output, 'w') as ofile:
        for iframe in range(nframes):
            natoms = 0
            elements = []
            geometries = []
            ids = []
            for n in range(len(files)):
                latoms = len(allgeos[n]['elements'][iframe])
                elements += allgeos[n]['elements'][iframe]
                geometries += allgeos[n]['geometries'][iframe]
                natoms += latoms
                ids += [n]*latoms
            ofile.write("%d\n" % natoms)
            ofile.write("NFrame = %d\n" % n)
            for iatom in range(natoms):
                values = (elements[iatom], geometries[iatom][0],
                          geometries[iatom][1], geometries[iatom][2],
                          ids[iatom])
                ofile.write("%s\t%.8f\t%.8f\t%.8f\t%d\n" % values)
