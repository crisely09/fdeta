
# Import package, test suite, and other packages as needed
import os
import numpy as np
import pytest
from nose.tools import assert_raises

from fdeta.units import BOHR
from fdeta.traj_tools import data_from_file
from fdeta.traj_tools import read_gromacs_trajectory
from fdeta.mdtrajectory import get_charge_dist, get_charge_and_electron_dist


def test_get_charge_distributions():
    """Test the functions to evaluate charge distributions (from histograms)"""
    dcharges = {'O': -0.834, 'H': 0.417}
    atoms = ['O', 'H', 'H', 'H', 'H', 'O']
    charges = np.array([dcharges[atom] for atom in atoms])
    geometry = np.array([[0.0, 0.0, 1.398901],
                         [3.4550165025, -0.7887247211, -2.5232747407],
                         [3.3481723623, 0.2098734901, -1.3908565749],
                         [-2.6344446631, -1.4682423133, -2.90767675],
                         [-2.0277667645, -2.8507288072, -2.799704616],
                         [3.3002789811, 0.1386994352, -2.3441575336]])

    grid_shape = np.array([[-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0]])
    grid_bins = (20, 20, 20)
    zcharge, charge_grid = get_charge_dist(atoms, geometry, charges, grid_shape, grid_bins)
    ref_dcharges = dict(O=8.0, H=1.0)
    nonzero = charge_grid != 0
    assert np.sum(nonzero) == 6  # Check if only 6 charges found
    assert zcharge == ref_dcharges  # Check if only 2 unique charge/elements are found
    # Use elements with different charges, like carbons
    new_atoms = ['C', 'C', 'C']
    coords = np.array([[0.0, 0.0, 0.180602],
                       [-0.0, 1.28307, -0.610989],
                       [-0.0, -1.28307, -0.610989]])
    new_charges = np.array([0.6, -0.2, 0.1])
    assert_raises(ValueError, get_charge_dist, new_atoms, coords, new_charges, grid_shape, grid_bins, zcharge)
    new_zcharge, new_charge_grid = get_charge_dist(new_atoms, coords, new_charges, grid_shape, grid_bins)
    nonzero = new_charge_grid != 0
    assert np.sum(nonzero) == 3
    added_zcharge, added_charge_grid = get_charge_dist(new_atoms, coords, new_charges, grid_shape,
                                                       grid_bins, zcharge, charge_grid)
    nonzero = added_charge_grid != 0
    assert np.sum(nonzero) == 9
    assert np.sum(added_charge_grid == 0.6)
    assert np.sum(added_charge_grid == 0.1)
    assert np.sum(added_charge_grid == -0.2)


def test_get_charge_and_electron_distributions():
    """Test the functions to evaluate electron distributions (from histograms)"""
    dcharges = {'O': -0.834, 'H': 0.417}
    atoms = ['O', 'H', 'H', 'H', 'H', 'O']
    charges = np.array([dcharges[atom] for atom in atoms])
    geometry = np.array([[0.0, 0.0, 1.398901],
                         [3.4550165025, -0.7887247211, -2.5232747407],
                         [3.3481723623, 0.2098734901, -1.3908565749],
                         [-2.6344446631, -1.4682423133, -2.90767675],
                         [-2.0277667645, -2.8507288072, -2.799704616],
                         [3.3002789811, 0.1386994352, -2.3441575336]])

    grid_shape = np.array([[-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0]])
    grid_bins = (20, 20, 20)
    zcharge, charge_grid, electron_grid = get_charge_and_electron_dist(atoms, geometry, charges, grid_shape, grid_bins)
    oxygen = 8.0 + 0.834
    hydrogen = 1.0 - 0.417
    assert np.sum(charge_grid) == 0.0
    assert np.sum(electron_grid == oxygen) == 2
    assert np.sum(electron_grid == hydrogen) == 4
    new_atoms = ['C', 'C', 'C']
    coords = np.array([[0.0, 0.0, 0.180602],
                       [-0.0, 1.28307, -0.610989],
                       [-0.0, -1.28307, -0.610989]])
    new_charges = np.array([0.6, -0.2, 0.1])
    assert_raises(ValueError, get_charge_and_electron_dist, new_atoms, coords, new_charges, grid_shape,
                  grid_bins, zcharge)
    assert_raises(ValueError, get_charge_and_electron_dist, new_atoms, coords, new_charges, grid_shape,
                  grid_bins, zcharge, charge_grid)
    new_zcharge, new_charge_grid, new_electron_grid = get_charge_and_electron_dist(new_atoms, coords, new_charges,
                                                                                   grid_shape, grid_bins)
    nonzero = new_charge_grid != 0
    assert np.sum(nonzero) == 3
    print('sum', np.sum(new_charge_grid))
    assert abs(np.sum(new_charge_grid) - 0.5) < 1e-9 
    assert np.sum(new_electron_grid) == sum([6.0 - charge for charge in new_charges])
    added_zcharge, added_charge_grid, added_electron_grid = get_charge_and_electron_dist(new_atoms, coords, new_charges,
                                                                                         grid_shape, grid_bins, zcharge,
                                                                                         charge_grid, electron_grid)
    assert abs(np.sum(added_charge_grid) - 0.5) < 1e-9
    diff = np.sum(new_electron_grid) == sum([6.0 - charge for charge in new_charges]) + 2*oxygen + 2*hydrogen
    assert diff < 1e-8


if __name__ == "__main__":
    test_get_charge_distributions()
    test_get_charge_and_electron_distributions()
