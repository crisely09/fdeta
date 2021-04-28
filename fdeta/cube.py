#!usr/bin/env python3
"""Some tools to handle cubefile format.

By C.G.E., Feb. 2020
Based on cubetools:
https://github.com/NicoRicardi/cubetools/blob/master/cubetools.py
from Niccolò Ricardi
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
               fname: str, comment: str = None,
               gauss_style: bool = False):
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
    gauss_style : bool
        Whether to print the values at each point as in Gaussian
        (5 values per line). False option prints each value per line.

    """
    if comment is None:
        comment = default_comment
    natoms = len(atoms)
    head = "{:5}{:12.6f}{:12.6f}{:12.6f}\n"
    satoms = "{:5}{:12.6f}{:12.6f}{:12.6f}{:12.6f}\n"
    if atoms.dtype == '<U2' or atoms.dtype == '<U1':
        numbers = [pt.to_atomic_number(a) for a in atoms]
    elif atoms.dtype == 'int':
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
            if gauss_style:
                if (n+1) % 6 == 0 or n == len(values)-1:
                    output.write("{:12.6e}\n".format(value))
                else:
                    output.write("{:12.6e} ".format(value))
            else:
                output.write("{:12.6e}\n".format(value))


def make_cubic_grid(grid_shape: tuple, vectors: np.ndarray,
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
        beg = origin[i]
        end = beg + steps*size
        vector = np.arange(beg, end, size)
        lvec = len(vector)
        if lvec != steps:
            if lvec < steps:
                vector = np.arange(beg, end+size, size)
            else:
                vector = np.arange(beg, end-size, size)
        axis.append(vector)
    return make_cube_from_points(axis)


def make_cube_from_points(points):
    """From the points in each direction make cubic grid.

    Parameters
    ----------
    points : np.ndarray
        Points on each direction that make the grid

    Returns
    -------
    grid3d :  np.ndarray((npoints, 3))
        Final 3D grid with npoints = N1*N2*N3
        where Ns are defined by the grid_shape.
    """
    # This swap of x and y is needed because of the z, y, x
    # evolution of cubic grids
    xv, yv, zv = np.meshgrid(points[1], points[0], points[2])
    xv = xv.reshape((xv.size,))
    yv = yv.reshape((yv.size,))
    zv = zv.reshape((zv.size,))
    ziplist = list(zip(yv, xv, zv))
    grid3d = np.array(ziplist)
    return grid3d


def make_grid_from_data(data: dict):
    """Make a cubic grid from the grid data.

    Parameters
    ----------
    data : dict
        Information of the data on cubefile format.

    Returns
    -------
    grid3d : np.ndarray((N, 3))
        3D cubic grid.
    """
    grid_shape = data['grid_shape']
    vectors = data['vectors']
    origin = data['origin']
    return make_cubic_grid(grid_shape, vectors, origin)


def check_data_same_cube(data0, data1):
    """Check if two sets of data from cubicfiles contain
       the same grid.

    Parameters
    ----------
    data0, data1 :  dict
        Information of the cubefiles.

    Raises
    ------
    ValueError:
        When data does not contain exact same information
    """
    if data0['elements'] != data1['elements']:
        raise ValueError('`elements` of each cubefile are different.')
    if not np.allclose(data0['coords'], data1['coords']):
        raise ValueError('`coords` of each cubefile are different.')
    if data0['grid_shape'] != data1['grid_shape']:
        raise ValueError('`grid_shape` of each cubefile are different.')
    if not np.allclose(data0['origin'], data1['origin']):
        raise ValueError('`origin` of each cubefile are different.')
    if not np.allclose(data0['vectors'], data1['vectors']):
        raise ValueError('`vectors` of each cubefile are different.')


def make_supercube(blims, atoms, geometries):
    """Expand the box into each direction.

    Parameters
    ----------
    blims : np.array
        Limits of the box in each axis
    atoms : list/array (str)
        Atoms symbols.
    geometries : np.ndarray
        Coordinates of all molecules/atoms.
    """
    latoms = len(geometries)
    tot_len = 27*latoms
    new_geometries = np.zeros((tot_len, 3))
    new_atoms = [atom for atom in 27*list(atoms)]
    for cube in range(27):
        start = cube*latoms
        fin = start + latoms
        new_geometries[start:fin] = geometries.copy()
    sizes = [blims[axis, 1] - blims[axis, 0] for axis in range(3)]
    signs = [-1, 0, 1]
    # Expand in each axis
    for axis0 in range(3):
        for axis1 in range(3):
            for axis2 in range(3):
                acount = axis2*latoms + 3*axis1*latoms + (9*axis0*latoms)
            #   print(axis0, axis1, axis2, acount)
                fin = acount + latoms
                new_geometries[acount:fin, 0] += signs[axis0]*sizes[axis0]
                new_geometries[acount:fin, 1] += signs[axis1]*sizes[axis1]
                new_geometries[acount:fin, 2] += signs[axis2]*sizes[axis2]
    return new_atoms, new_geometries


def expand_grid_information(values_hist):
    """Return the expanded values on each grid point.

    Note
    ----
    Used with array/tuple from np.histogramdd.

    Parameters
    ----------
    values_hist
    """
    vshape = values_hist.shape
    nx = vshape[0]
    ny = vshape[1]
    nz = vshape[2]
    result = np.zeros(nx*ny*nz)
    values = values_hist.flatten()
    count = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                fcount = z + y*ny + x*ny*nz
                result[count] = values[fcount]
                count += 1
    return result


def box_for_molecule(mol_arr, margin=2.0):
    """
    Parameters
    ----------
    arr: np.ndarr(natoms, 3)
        Geometry (object of filename).
    margin: float or in
        The margin from the extreme nuclei.
    Returns
    -------
    array(3,2)
        [[i.min,i.max] for i in x,y,z]
    """
    return np.array([mol_arr.min(axis=0)-margin, mol_arr.max(axis=0)+margin]).T


def grid_from_box(box, dist=0.2, fix="box"):
    """
    Parameters
    ----------
    box: array(3,2)
        [[i.min, i.max] for i in x,y,z] 
    dist: int or float
        distance between points
    fix: str
        "box" and some others keep box unchanged and slightly reduce the distance
        "dist" and some others keep the distance unchanged and slightly increase the box size
    Returns
    -------
    tuple(grid_shape, steps, origin)
        Vector of number of points, Vector size matrix, origin
    """
    box_size = box.ptp(axis=1)
    grid_shape = np.divide(box_size, dist)+1
    from math import ceil
    grid_shape = np.array([ceil(i) for i in grid_shape])
    origin = box[:, 0]
    if fix in ["box","extremes","margins"]:  # box is kept but the voxel is reduced slightly
        steps = np.divide(box_size, grid_shape)
        print("Your voxel now has size {},{},{}".format(*np.divide(box_size, grid_shape)))
    if fix in ["dist", "distance", "voxel", "vect", "vector", "maxdist", "max_dist"]:
        steps = 3*[dist]
        box[:, 1] = box[:, 0] + dist * grid_shape
        print("Your box has been changed to: ({},{}),({},{}),({},{})".format(*box.reshape(-1)))
    return (grid_shape, steps, origin)


def cut_cube_geometries(limits, atoms, geometries, natoms=None,
                        extra_space=0.10):
    """Cut a set of geometries to keep the ones inside a cube.

    Parameters
    ----------
    limits : list/array(float)
        The limits for the cube coordinates [min, max].
    atoms : list/array(str)
        Atom symbols.
    geometries : np.ndarray(N, 3)
        Coordinates of the atoms/molecules of the system.
    natoms : dict
        The number of atoms of each type (keys are atomic symbols).
    extra_space : float
        Extra space to add to cube limits to include more atoms
    """
    if isinstance(atoms, list):
        tmp_atoms = np.array(atoms)
    else:
        tmp_atoms = atoms.copy()
    # If we do not care about conserving the number of atoms.
    if natoms is None:
        inside_geos = []
        for axis in range(3):
            inside_geos.append(np.logical_and(geometries[:, axis] > limits[0], limits[1] > geometries[:, axis]))
        zipped = list(zip(inside_geos[0], inside_geos[1], inside_geos[2]))
        zipped = np.array([list(inner) for inner in zipped], dtype=bool)
        inside = np.where(zipped.all(axis=1))[0]
        new_atoms = tmp_atoms[inside]
        new_geometries = geometries[inside] 
        return new_atoms, new_geometries
    # If we want to kee a number of atoms fixed.
    else:
        # Add extra space to keep more atoms
        new_limits = limits.copy()
        new_limits[0] -= extra_space
        new_limits[1] += extra_space
        inside_geos = []
        for axis in range(3):
            inside_geos.append(np.logical_and(geometries[:, axis] > new_limits[0], new_limits[1] > geometries[:, axis]))
        zipped = list(zip(inside_geos[0], inside_geos[1], inside_geos[2]))
        zipped = np.array([list(inner) for inner in zipped], dtype=bool)
        inside_tmp = np.where(zipped.all(axis=1))[0]
        inside = []
        # Check atom by atom if numbers are respected from the beginning
        geos = geometries[inside_tmp]
        a = tmp_atoms[inside_tmp]
        for atom in natoms:
            atom_indices = np.where(a == atom)[0]
            asum = len(atom_indices) 
            if asum < natoms[atom]:
                raise ValueError('`extra_space` is not enough to count more atoms.(%d, %d)' % (asum, natoms[atom]))
            excess = asum - natoms[atom]
            print('excess', excess)
            atom_geos = geos[atom_indices]
            eds = []
            ds_from_limits = []
            # Get distance excess for each atom
            for gatom in atom_geos:
                # Sum up the excess in each axis
                tmp_eds = 0.0
                downexcess = np.where(gatom < limits[0])[0]
                upexcess = np.where(gatom > limits[1])[0]
                if len(upexcess) >= 1:
                    for u in upexcess:
                        tmp_eds += gatom[u] - limits[1] 
                if len(downexcess) >= 1:
                    for u in downexcess:
                        tmp_eds += limits[0] - gatom[u]
                eds.append(tmp_eds)
                # Also save the total distance from the limits
                tmp_dfl = 0.0
                for axis in range(3):
                    if gatom[axis] < limits[0] + (limits[1]/2.0):
                        tmp_dfl += abs(gatom[axis]) - abs(limits[0])
                    else:
                        tmp_dfl += abs(gatom[axis]) - abs(limits[1])
                ds_from_limits.append(tmp_dfl)
            # Keep the atoms with smallest or non excess
            non_zero = [i for i in range(len(atom_indices)) if eds[i] > 0.0]
            if len(non_zero) < excess:
                more_excess = excess - len(non_zero)
                # Include indices from the 
                dfl_ordered = sorted(ds_from_limits, reverse=True)
                for dfl in dfl_ordered:
                    if more_excess > 0:
                        if ds_from_limits.index(dfl) not in non_zero:
                            new_index = ds_from_limits.index(dfl)
                            non_zero.append(new_index)
                            more_excess -= 1
                    else:
                        break
            elif excess < len(non_zero):
                less_excess = len(non_zero) - excess
                eds_ref = [eds[i] for i in non_zero]
                eds_ordered = sorted(eds_ref, reverse=True)
                off = []
                for i in range(less_excess):
                    off.append(eds_ref.index(eds_ordered[i]))
                rlen = len(non_zero)
                non_zero = [non_zero[ix] for ix in range(rlen) if ix not in off]
            inside += [atom_indices[j] for j in range(len(atom_geos)) if j not in non_zero]
        return a[inside], geos[inside]
