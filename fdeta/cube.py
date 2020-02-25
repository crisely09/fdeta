#!usr/bin/env python3
"""Some tools to handle cubefile format.

By C.G.E., Feb. 2020
"""

import numpy as np
from typing import Union
from qcelemental import periodictable as pt


default_comment = """  CUBE FILE.
  OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n"""


def read_cubefile(fname: str):
    """ Read cubefile.
    (Based on Nico's code)

    Parameters
    ----------
    fname: str
        Filename or path.

    Returns
    -------
    cube : dict
        All the information: atoms, coords,

    """
    nsteps = []
    vectors = []
    comment = ""
    natm = 1
    positions = []
    atoms = []
    npoints = 1
    with open(fname, 'r') as f:
        cube = f.read()
    lines = cube.splitlines()
    for j, line in enumerate(lines):
        if j < 2:
            comment += line
        elif j == 2:
            splt = line.split()
            natm = int(splt[0])
            origin = np.array([float(p) for p in splt[1:]])
        elif j > 2 and j < 6:
            splt = line.split()
            nsteps.append(int(splt[0]))
            npoints *= nsteps[-1]
            vectors.append(np.array(list(map(float, splt[1:4]))))
        elif j >= 6 and j < natm + 6:
            splt = line.split()
            atoms.append(pt.to_symbol(int(splt[0])))
            positions.append(np.array(list(map(float, splt[2:5]))))
        else:
            break
    grid_shape = tuple(nsteps)
    vectors = np.array(vectors)
    coords = np.array(positions)
    atoms = np.array(atoms)
    if len(lines[natm+6:]) == npoints:
        values = [float(l.split()[0]) for l in lines[natm+6:]]
    else:
        values = []
        for line in lines[natm+6:]:
            for e in line.split():
                values.append(float(e))
    values = np.array(values)
    cube = dict(atoms=atoms, coords=coords, origin=origin,
                vectors=vectors, grid_shape=grid_shape,
                values=values)
    return cube


def write_cube(atoms: np.ndarray, coords: np.ndarray,
               origin: np.ndarray, vectors: np.ndarray,
               gridshape: tuple, values: Union[list, np.ndarray],
               fname: str, comment: str = None):
    """Write cubefile in Gaussian format.

    Parameters
    ----------
    atoms : np.ndarray
        Atoms of the molecule, atomic symbols.
    coords :  np.ndarray((natoms, 3), dtype=float)
        Spacial coordinates of atoms, in Bohr.
    origin : np.ndarray((3,), dtype=float)
        Where to place the origin of the axis.
    gridshape : tuple(3)
        Number of steps performed in each direction (nx, ny, nz).
    values : np.ndarray(dtype=float)
        Array with all the values arranged so z moves first, then y
        then x.
    comment :  str
        First two lines of the cubefile.

    """
    if comment is None:
        comment = default_comment
    natoms = len(atoms)
    head = "{:5}{:12.6f}{:12.6f}{:12.6f}\n"
    satoms = "{:5}{:12.6f}{:12.6f}{:12.6f}{:12.6f}\n"
    if atoms.dtype == '<U1':
        numbers = [pt.to_atomic_number(a) for a in atoms]
    elif atoms.dtype == 'int64':
        numbers = atoms
    else:
        raise TypeError("`atoms` must be provided as str or int.")
    with open(fname, "w") as output:
        output.write(comment)
        output.write(head.format(natoms, origin[0], origin[1], origin[2]))
        for i in range(3):
            output.write(head.format(gridshape[i], vectors[i, 0],
                                     vectors[i, 1], vectors[i, 2]))
        for i in range(natoms):
            output.write(satoms.format(numbers[i], 0.0, coords[i, 0],
                                       coords[i, 1], coords[i, 2]))
        for n, value in enumerate(values):
            if (n+1) % 6 == 0 or n == len(values)-1:
                output.write("{:12.6e}\n".format(value))
            else:
                output.write("{:12.6e} ".format(value))


def make_grid(grid_shape: tuple, vectors: np.ndarray,
              origin: np.ndarray):
    """Make 3D grid from cube specifications.

    Parameters
    ----------
    grid_shape : tuple(int)
        Shape of final 3D grid.
    vectors : np.ndarray((3,3) dtype=float)
        Steps taken in each direction.
    origin : np.ndarray((3,) dtype=float)
        Origin, where the grid is built from.

    Returns
    -------
    grid3d :  np.ndarray((npoints, 3))
        Final 3D grid with npoints = N1*N2*N3
        where Ns are defined by the grid_shape.
    """
    axis = []
    for i in range(3):
        steps = grid_shape[i]
        size = vectors[i, i]
        axis.append(np.arange(origin[i], steps*size, size))

    xv, yv, zv = np.meshgrid(axis[0], axis[1], axis[2])
    xv = xv.reshape((xv.size,))
    yv = yv.reshape((yv.size,))
    zv = zv.reshape((zv.size,))
    ziplist = list(zip(yv, xv, zv))
    grid3d = np.array(ziplist)
    return grid3d
