"""Set of tests for traj_tools.py"""

# Import package, test suite, and other packages as needed
import os
import pytest
import numpy as np

from fdeta.traj_tools import compute_center_of_mass, atom_to_mass
from fdeta.traj_tools import atom_to_charge, find_unique_elements
from fdeta.traj_tools import clean_atom_name, get_data_lines
from fdeta.traj_tools import default_charges, flatten_list
from fdeta.traj_tools import get_data_lines, read_xyz_trajectory
from fdeta.traj_tools import read_pqr_trajectory, read_gromacs_trajectory
from fdeta.traj_tools import data_from_file, make_trajectory_file


def test_compute_center_of_mass():
    """Function to compute center of mass."""
    # Water coordinates
    coords = np.array([[-2.62712, 0.94888, 0.00000],
                       [-1.65712, 0.94888, 0.00000],
                       [-2.95045, 1.75403, 0.43369]])
    masses = np.array([15.999, 1.008, 1.008])
    ref = np.array([[-2.5909, 0.9939, 0.0243]])
    com = compute_center_of_mass(masses, coords)
    assert np.allclose(ref, com, atol=1e-3)


def test_atom_to_mass():
    """Function to get atomic masses."""
    atoms = [1, 6, 4]
    ref = np.array([1.008, 12.0, 9.0122])
    final = np.array([atom_to_mass(a) for a in atoms])
    assert (abs(ref - final) < 1e-3).all()


def test_atom_to_charge():
    """Function to get atomic nuclear charges."""
    atoms = ['O', 'Ca', 'Br']
    ref = [8.0, 20.0, 35.0]
    final = [atom_to_charge(a) for a in atoms]
    assert np.allclose(ref, final)


def test_default_charges():
    """Function to give all charges of a list"""    
    atoms = ['O', 'Ca', 'Br']
    ref = [8.0, 20.0, 35.0]
    charges = default_charges(atoms)
    assert isinstance(charges, dict)
    final = [charges[e] for e in atoms]
    assert np.allclose(ref, final)


def test_flatten_list():
    """Function to flatten a ND list."""
    lst1 = ['Ca', 'O', [['H', 'Li', 'Be2'], 'HA']]
    lst2 = ['Ca', ['O', 'H'], ['Li', 'Be2', 'HA']]
    flt1 = flatten_list(lst1) 
    flt2 = flatten_list(lst2) 
    assert flt1 == ['Ca', 'O', 'H', 'Li', 'Be2', 'HA']
    assert flt1 == flt2


def test_find_unique_elements():
    """Function to identify unique element names."""
    atoms = ['Ca', 'O', 'H', 'H', 'Li', 'O', 'H', 'Be2',
              'HA']
    ref = ['Ca', 'O', 'H', 'Li', 'Be2', 'HA']
    uniques = find_unique_elements(atoms)
    assert ref.sort() == uniques.sort()


def test_clean_atom_name():
    """Function to delete atom index name from trajectory."""
    atoms = ['H', 'He1', 'HA', 'OW1', 'HW', 'CA', 'CE']
    heteroatoms = ['Ca', 'He']
    fin = [clean_atom_name(atom, heteroatoms) for atom in atoms]
    ref = ['H', 'He', 'H', 'O', 'H', 'C', 'C']
    assert fin == ref


def test_get_data_lines():
    """Function to read file or files."""
    with pytest.raises(TypeError):
        get_data_lines(0)
    with pytest.raises(TypeError):
        get_data_lines([0, 1])


def test_read_xyz_trajectory():
    """Function to read `.xyz` and `.fde` files."""
    ref_elements = ['C', 'C', 'C', 'H', 'H', 'H', 'H',
                    'H', 'H', 'O', 'H', 'H', 'H', 'H',
                    'O', 'O']
    ref_coords = np.array([[0.0, 0.0, 0.180602,],
                           [-0.0, 1.28307, -0.610989],
                           [-0.0, -1.28307, -0.610989]])
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.fde')
    geos1 = read_xyz_trajectory(traj)
    geos2 = read_xyz_trajectory(traj, has_ids=True)
    assert not flatten_list(geos1['elements'])
    assert geos2['elements'][0] == ref_elements
    assert np.allclose(geos2['geometries'][0][:3], ref_coords)
    assert geos2['ids'][0][:10] == ['0']*10


def test_read_pqr_trajectory():
    """Function to read `.pqr` files."""
    return


def gromacs_trajectory():
    """Function to read gromacs files."""
    dic = os.getenv('FDETADATA')
    fgro = os.path.join(dic, 'ace_mdft_spce.gro')
    ftrr = os.path.join(dic, 'ace_mdft_spce.trr')
    files = [fgro, ftrr]
    solute = range(10)
    data = read_gromacs_trajectory(files, solute)


def test_data_from_file():
    """Function to read `.xyz`, `.pqr`,  and `.fde` files."""
    with pytest.raises(ValueError):
        data_from_file('traj.gro')
    ref_elements = ['C', 'C', 'C', 'H', 'H', 'H', 'H',
                    'H', 'H', 'O', 'H', 'H', 'H', 'H',
                    'O', 'O']
    ref_coords = np.array([[0.0, 0.0, 0.180602,],
                           [-0.0, 1.28307, -0.610989],
                           [-0.0, -1.28307, -0.610989]])
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.fde')
    geos = data_from_file(traj)
    assert geos['elements'][0] == ref_elements
    assert np.allclose(geos['geometries'][0][:3], ref_coords)
    assert geos['ids'][0][:10] == ['0']*10


def test_make_trajectory_file():
    with pytest.raises(NotImplementedError):
        read_gromacs_trajectory('something', input_format='.gro')


def test_check_length_trajectories():
    return


def test_combine_fragment_files():
    return


if __name__ == "__main__":
    test_compute_center_of_mass()
    test_atom_to_mass()
    test_atom_to_charge()
    test_default_charges()
    test_flatten_list()
    test_find_unique_elements()
    test_clean_atom_name()
    test_read_xyz_trajectory()
    test_data_from_file()
#   test_gromacs_trajectory()
