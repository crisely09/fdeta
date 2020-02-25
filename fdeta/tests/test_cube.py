#!/usr/bin/env python3

import os
import numpy as np
from fdeta.fdetmd.cgrid_tools import BoxGrid
from fdeta.cube import read_cubefile, make_grid
from fdeta.cube import write_cube
from qcelemental import periodictable as pt


def test_write_and_read():
    dic = os.getenv('FDETADATA')
    name = 'solvent_charge_density_aroud_acetone.cube'
    fname = os.path.join(dic, name)
    data = read_cubefile(fname)
    grid_shape = data["grid_shape"]
    vectors = data["vectors"]
    origin = data["origin"]
    atoms = data["atoms"]
    coords = data["coords"]
    values = data["values"]
    numbers = np.array([pt.to_atomic_number(a) for a in atoms])
    fname = "solvent_new.cube"
    write_cube(numbers, coords, origin, vectors, grid_shape, values,
               fname)
    data2 = read_cubefile('solvent_new.cube')
    assert (data['atoms'] == data2['atoms']).all()
    assert np.allclose(data['coords'], data2['coords'])
    assert np.allclose(data['origin'], data2['origin'])
    assert np.allclose(data['vectors'], data2['vectors'])
    assert data['grid_shape'] == data2['grid_shape']
    os.remove('solvent_new.cube')


def test_grid_order():
    dic = os.getenv('FDETADATA')
    name = 'solvent_charge_density_aroud_acetone.cube'
    fname = os.path.join(dic, name)
    data = read_cubefile(fname)
    grid_shape = data["grid_shape"]
    vectors = data["vectors"]
    origin = data["origin"]
    charge = data["values"]
    grid3d = make_grid(grid_shape, vectors, origin)
    axis = []
    for i in range(3):
        steps = grid_shape[i]
        size = vectors[i, i]
        axis.append(np.arange(origin[i], steps*size, size))
    axis = np.array(axis).reshape((3, steps))
    shape = np.array(grid_shape)
    bgrid = BoxGrid(shape, axis)
    gridb = np.zeros((len(charge),3))
    gridb = bgrid.get_grid(gridb, True)
    assert np.allclose(grid3d, gridb)


if __name__ == "__main__":
    test_write_and_read()
    test_grid_order()
