
# Import package, test suite, and other packages as needed
import os
import numpy as np
import pytest

from fdeta.traj_tools import data_from_file
from fdeta.traj_tools import read_gromacs_trajectory
from fdeta.mdtrajectory_new import UElement, MDTrajectory, get_unique_elements


def test_uelement():
    element = 'C1'
    charge = 0.6
    efake = 0
    chfake = 'le'
    with pytest.raises(TypeError):
        uelement = UElement(efake, charge)
    with pytest.raises(TypeError):
        uelement = UElement(element, chfake)
    uelement = UElement(element, charge)
    assert uelement.symbol == 'C'
    assert uelement.zcharge == 6
    assert abs(uelement.mass - 12.0) < 1e-2


def test_get_unique_elements():
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.fde')
    data = data_from_file(traj)
    elements = data['elements']
    charges = {'C':0.6, 'O': -0.834, 'H': 0.417}
    uniques = get_unique_elements(elements, charges)
    assert len(uniques) == 3
    # Carbon
    assert uniques[0].name == 'C'
    assert uniques[0].symbol == 'C'
    assert uniques[0].charge == 0.6
    assert uniques[0].alloc_traj[0] == uniques[0].alloc_traj[1]
    assert uniques[0].count_frames[0] == uniques[0].count_frames[1]
    assert uniques[0].total_count == 6
    # Hydrogen
    assert uniques[1].name == 'H'
    assert uniques[1].symbol == 'H'
    assert uniques[1].charge == 0.417
    assert uniques[1].alloc_traj[0] == uniques[1].alloc_traj[1]
    assert uniques[1].count_frames[0] == uniques[1].count_frames[1]
    assert uniques[1].total_count == 20
    # Oxygen
    assert uniques[2].name == 'O'
    assert uniques[2].symbol == 'O'
    assert uniques[2].charge == -0.834
    assert uniques[2].alloc_traj[0] == uniques[2].alloc_traj[1]
    assert uniques[2].count_frames[0] == uniques[2].count_frames[1]
    assert uniques[2].total_count == 6


def test_base():
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.fde')
    data = data_from_file(traj)
    elements = [elems[10:] for elems in data['elements']]
    coordinates = [coords[10:] for coords in data['geometries']]
    charges = {'O': -0.834, 'H': 0.417}
    ta = MDTrajectory(elements, coordinates, charges)
    assert len(ta.unique_elements) == 2

#   edges, pcf = ta.compute_pair_correlation_function(histogram_range, grid_size, 0)
#   assert len(np.where(pcf['O'] > 0)[0]) == 4
#   assert len(np.where(pcf['H'] > 0)[0]) == 8
#   # Test saving correct geometries
#   h2o_iframe = ta.get_structure_from_trajectory(1, 1, ta.trajectory)
#   h2o_wrong = ta.get_structure_from_trajectory(1, 0, ta.trajectory)
#   ref = np.array([[0.823616466, 2.2301171341, -3.0272218672],
#          [0.8802243553, 0.8879174165, -3.7245599666],
#          [0.254821451, -1.3338434534, -5.1694547009],       
#          [1.2650984606, -1.5274583661, -4.0591220527],        
#          [1.2041949721, 1.7865317894, -3.7852022645],
#          [0.4788450611, -1.0324072555, -4.2890616746]])
#   assert np.allclose(h2o_iframe, ref)
#   assert np.allclose(h2o_wrong, ref)


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
    test_uelement()
    test_get_unique_elements()
    test_base()
#   test_files()
#   test_pcf()
