
# Import package, test suite, and other packages as needed
import os
import numpy as np

from fdeta.traj_tools import data_from_file
from fdeta.traj_tools import read_gromacs_trajectory
from fdeta.mdtrajectory import MDTrajectory


def test_base():
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.fde')
    box_size = np.array([10, 10, 10])
    grid_size = np.array([8, 8, 8])
    histogram_range = np.asarray([-box_size/2., box_size/2.]).T
    data = data_from_file(traj)
    ta = MDTrajectory(data)
    edges, pcf = ta.compute_pair_correlation_function(histogram_range, grid_size, 0)
    assert len(np.where(pcf['O'] > 0)[0]) == 4
    assert len(np.where(pcf['H'] > 0)[0]) == 8


def test_files():
    """Basic checks for MDTrajectory class."""
    dic = os.getenv('FDETADATA')
    ftraj = os.path.join(dic, 'he_traj.fde')
    data = data_from_file(ftraj)
    fcharges = os.path.join(dic, 'he_charges.txt')
    traj = MDTrajectory(data, fcharges)
    structure = traj.get_structure_from_trajectory(0, 0, traj.trajectory)
    assert np.allclose(structure, [0., 0., 0.])
    traj.save_trajectory(0)
    with open('trajectory_0.fde', 'r') as tf:
        text = tf.read()
    reftext = """1\nFrame 0\nHe 0.0 0.0 0.0 0\n"""
    assert text == reftext
    os.remove('trajectory_0.fde')
    # save average
    traj.get_average_structure(0, 'coords')
    with open('snapshot_0.fde', 'r') as tf:
        text = tf.read()
    assert text == reftext


def test_pcf():
    """Test if pcf works"""
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.fde')
    data = data_from_file(traj)
    ta = MDTrajectory(data)
    box_size = np.array([6, 6, 6])
    grid_size = np.array([10, 10, 10])
    histogram_range = np.asarray([-box_size/2., box_size/2.]).T
    edges, pcf = ta.compute_pair_correlation_function(histogram_range,
                                                      grid_size, 0)
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
    test_files()
    test_pcf()
