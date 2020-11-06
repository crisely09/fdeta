
# Import package, test suite, and other packages as needed
import os
import numpy as np
import pytest

from fdeta.units import BOHR
from fdeta.traj_tools import data_from_file
from fdeta.traj_tools import read_gromacs_trajectory
from fdeta.mdtrajectory import MDTrajectory


def test_base():
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.fde')
    data = data_from_file(traj)
    elements = [elems[10:] for elems in data['elements']]
    coordinates = [coords[10:] for coords in data['geometries']]
    charges = {'O': -0.834, 'H': 0.417}
    ta = MDTrajectory(elements, coordinates, charges)
    assert len(ta.unique_elements) == 2
    # Check PCFs
    box_size = np.array([11, 11, 11])
    grid_range = np.array([-box_size/2., box_size/2.]).T
    grid_bins = (8, 8, 8)
    pcf = ta.compute_pair_correlation_functions(grid_range, grid_bins)
    assert len(np.where(pcf['O'] > 0)[0]) == 4
    assert len(np.where(pcf['H'] > 0)[0]) == 8
    assert ta.unique_elements[0].frame_count == 2
    assert ta.unique_elements[1].frame_count == 2
    charge_density, electron_density = ta.compute_charge_densities()
    assert abs(sum(charge_density) - 0.0) < 1e-11
    dv = (box_size[0]/grid_bins[0])**3
    # Check the total electron count is consistent, 2 water = 20 electrons
    assert abs(sum(electron_density)*dv/BOHR**3 - 20.0) < 1e-11


def test_pcf():
    """Test if pcf works"""
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.fde')
    data = data_from_file(traj)
    charges = {'O': -0.834, 'C': 0.2, 'H': 0.417}
    ta = MDTrajectory(data['elements'], data['geometries'], charges)
    box_size = np.array([6, 6, 6])
    grid_size = np.array([10, 10, 10])
    histogram_range = np.asarray([-box_size/2., box_size/2.]).T
    pcf = ta.compute_pair_correlation_functions(histogram_range,
                                                      grid_size)
    pcfO = np.where(pcf["O"] == 1.0)
    assert np.allclose(pcfO, (np.array([1]), np.array([1]), np.array([0])))


def gromacs():
    """Test if pcf works"""
    dic = os.getenv('FDETADATA')
    fgro = os.path.join(dic, 'ace_mdft_spce.gro')
    ftrr = os.path.join(dic, 'ace_mdft_spce.trr')
    files = [fgro, ftrr]
    solute = range(10)
    data = read_gromacs_trajectory(files, solute)
    ta = MDTrajectory(data)


if __name__ == "__main__":
    test_base()
#   test_pcf()
#   test_files()
