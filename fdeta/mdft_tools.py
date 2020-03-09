# -*- coding: utf-8 -*-
#  by CGE, 2020.
"""
Tools for connectivity with Molecular-DFT.

"""

import numpy as np
from typing import Union, List
from qcelemental import periodictable


def check_length(arrays: list):
    """Compare length of arrays.

    Parameters
    ----------
    arrays : list
        List of arrays to be compared.

    Raises
    ------
    ValueError if length of arrays don't match.
    """
    if len(arrays) < 2:
        raise ValueError("At least two arrays need to be given.")
    lref = len(arrays[0])
    for i, a in enumerate(arrays[1:]):
        if len(a) != lref:
            raise ValueError("Array in position %d has different length." % i)


def read_parameters_file(fname: str) -> dict:
    """Read parameters file.

    Parameters
    ----------
    fname : str
        Name of file with parameters.

    Returns
    -------
    data : dict
        All information obtained from file:
        indices, elements, surnames, coords, vcharge,
        vsigma, vepsilon.
    """
    indices = []
    vcharge = []
    vsigma = []
    vepsilon = []
    coords = []
    elements = []
    surnames = []
    with open(fname, 'r') as infile:
        lines = infile.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                comment = line
            elif i == 1:
                data = [int(e) for e in line.split()]
                if len(data) != 2:
                    raise ValueError("Wrong format in line 2, expected 2 integers.")
                natoms = data[0]
                uniques = data[1]
            elif i >= 3:
                data = line.split()
                if len(data) != 10:
                    raise ValueError("Wrong number of arguments in line %d, expected 10.", i)
                indices.append(int(data[0]))
                vcharge.append(float(data[1]))
                vsigma.append(float(data[2]))
                vepsilon.append(float(data[3]))
                coords.append([float(c) for c in data[4:7]])
                elements.append(data[8])
                surnames.append(data[9])
    if natoms != len(elements):
        raise ValueError("Number of atoms and lines don't match.")

    # Make all arrays
    indices = np.array(indices)
    elements = np.array(elements, dtype=str)
    surnames = np.array(surnames, dtype=str)
    vcharge = np.array(vcharge)
    vsigma = np.array(vsigma)
    vepsilon = np.array(vepsilon)
    coords = np.array(coords)
    output = dict(indices=indices, elements=elements, surnames=surnames, coords=coords,
                  vcharge=vcharge, vsigma=vsigma, vepsilon=vepsilon, comment=comment,
                  uniques=uniques)
    return output


def write_parameters_file(elements: Union[List, np.ndarray],
                          surnames: Union[List, np.ndarray],
                          coords: np.ndarray,
                          vcharge: Union[List, np.ndarray],
                          vsigma: Union[List, np.ndarray],
                          vepsilon: Union[List, np.ndarray],
                          fname: str = "pars.in",
                          comment: str = "Input made with FDETA module.\n"):
    """Write text file with parameters for MDFT.

    Parameters
    ----------
    elements : list or array of str
        Elements of the fragment or molecule.
    surnames : list or array of str
        Surnames for the elements of the fragment or molecule.
    coords : np.ndarray
        Cartesian coordinates in Angstrom of the fragment or molecule.
    vcharge : list or array [float]
        Vector with charges for each element.
    vsigma : list or array [float]
        Vector with sigma values (LJ parameters).
    vepsilon : list or array [float]
        Vector with epsilon values (LJ parameters).
    comment : str
        Comment to be added in the first line of the file.
    """
    # Check all arrays have the same lenght
    check_length([elements, surnames, coords, vcharge, vsigma, vepsilon])
    # Check for repeated and unique elements
    natoms = len(elements)
    repeated = []
    uniques = [None]*natoms
    count = 0
    for i in range(natoms):
        for j in range(i+1, natoms):
            # Assumes that vsigma and vepsilon are always paired (same value)
            if vcharge[i] == vcharge[j] and vsigma[i] == vsigma[j]:
                repeated += [i, j]
                if not uniques[j]:
                    uniques[j] = i + 1 - count
                count += 1
            else:
                if not uniques[i]:
                    if i not in repeated:
                        uniques[i] = i + 1 - count
                    else:
                        uniques[i] = i + 2 - count
    if not uniques[-1]:
        uniques[-1] = natoms - count
    nuniques = natoms - count
    title_format = '{:3} {:^10} {:^10} {:10} {:^13} {:^13} {:^13} {:^8} {:^10} {:^8}\n'
    header = title_format.format('#', 'charge', 'sigma', 'epsilon', 'x', 'y', 'z',
                                 'Z', 'Atom name', 'Surname')
    out = ""
    for i in range(natoms):
        out += '{:3}'.format(str(uniques[i]))
        out += '{:10.6f}'.format(vcharge[i])
        out += '{:10.6f}'.format(vsigma[i])
        out += '{:10.6f}'.format(vepsilon[i])
        for j in range(3):
            out += '{:>15.10f}'.format(coords[i, j])
        out += '{:>5}'.format(str(periodictable.to_atomic_number(elements[i])))
        out += '{:^19}'.format(elements[i])
        out += '{}\n'.format(surnames[i])
    # Write file
    with open(fname, 'w') as outfile:
        outfile.write(comment)
        outfile.write("%d  %d\n" % (natoms, nuniques))
        outfile.write(header)
        outfile.write(out)
