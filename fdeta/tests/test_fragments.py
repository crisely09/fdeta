"""
Unit and regression test for the fdeta package.

"""

# Import package, test suite, and other packages as needed
import os
import numpy as np
from nose.tools import assert_raises

from fdeta.fragments import get_bond_matrix, common_members
from fdeta.fragments import Fragment, Ensemble
from fdeta.fragments import _align_from_scratch, _align_from_matrices, align_frames
from fdeta.fragments import find_fragments, get_interfragment_distances
from fdeta.units import BOHR



def test_align_basics():
    """Test basic functions for alignment."""
    # Test align from scratch
    atoms = ['He', 'He']
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
    coords2 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 2.5]])
    geos = dict(atoms=[atoms, atoms], coords=[coords, coords2])
    ref_geo = dict(atoms=atoms, coords=coords)
    # Check types
    assert_raises(TypeError, _align_from_scratch, atoms, ref_geo)
    assert_raises(ValueError, _align_from_scratch, geos, geos)
    dic = os.getenv('FDETADATA')
    mpath = os.path.join(dic, 'test_path')
    _align_from_scratch(geos, ref_geo, save_matrices=True, mat_path=mpath)
    assert np.allclose(geos['coords'][0], ref_geo['coords'])
    assert np.allclose(geos['coords'][1], ref_geo['coords'])
    # Test align from matrices
    geos2 = dict(atoms=[atoms, atoms], coords=[coords, coords2])
    assert_raises(TypeError, _align_from_matrices, atoms, mpath)
    assert_raises(ValueError, _align_from_matrices, atoms, None)
    assert_raises(TypeError, _align_from_matrices, geos2, ref_geo)
    assert_raises(ValueError, _align_from_matrices, geos2, os.getcwd())
    _align_from_matrices(geos2, mat_path=mpath)
    assert np.allclose(geos2['coords'][0], ref_geo['coords'])
    assert np.allclose(geos2['coords'][1], ref_geo['coords'])


def test_align_frames():
    """Test the align function."""
    # Test align from scratch
    atoms = ['He', 'He']
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
    coords2 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 2.5]])
    geos = dict(atoms=[atoms, atoms], coords=[coords, coords2])
    ref_geo = dict(atoms=atoms, coords=coords)
    assert_raises(ValueError, align_frames, geos)
    align_frames(geos, ref_geo)
    assert np.allclose(geos['coords'][0], ref_geo['coords'])
    assert np.allclose(geos['coords'][1], ref_geo['coords'])


def test_fragment():
    """Test Fragment object."""
    atoms = ['H']
    coords = [[0.0, 0.1, 0.2]]
    charges = [0.1]
    assert_raises(TypeError, Fragment, 'K', coords, charges)
    assert_raises(TypeError, Fragment, atoms, 0.6, charges)
    assert_raises(TypeError, Fragment, atoms, coords, False)
    assert_raises(ValueError, Fragment, ['H', 'He'], coords)
    assert_raises(ValueError, Fragment, atoms, coords, [0.1, 0.2])


def test_fragment_from_file():
    """Test Fragment from_file method."""
    dic = os.getenv('FDETADATA')
    xyz = os.path.join(dic, 'he_test_0.xyz')
    pqr = os.path.join(dic, 'test_0.pqr')
    frag = Fragment.from_file(xyz)
    assert (frag.atoms == ['He', 'He']).all()
    assert np.allclose([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]], frag.coords)
    assert frag.charges is None
    frag = Fragment.from_file(pqr)
    assert (frag.atoms == ['H','C', 'H']).all()
    ref_coords = np.array([[25.512, 38.671, 49.092],
                           [26.263, 39.012, 49.834],
                           [25.781, 39.262, 50.817]])
    assert np.allclose(ref_coords, frag.coords)
    ref_charges = np.array([0.074, 0.028, 0.057])
    assert np.allclose(ref_charges, frag.charges)


def test_ensemble():
    """Test basic Ensemble object."""
    atoms = [['K', 'H', 'Na'], ['Mn']]
    coords = [np.array([[25.512, 38.671, 49.092],
                        [26.263, 39.012, 49.834],
                        [25.781, 39.262, 50.817]]),
              np.array([[0.0, 0.0, 0.0]])]
    charges = [[0.074, 0.028, 0.057], [0.1]]
    assert_raises(ValueError, Ensemble, ['K'], coords, charges)
    assert_raises(ValueError, Ensemble, atoms, [0.0, 0.1, 0.2], charges)
    assert_raises(ValueError, Ensemble, atoms, coords, [])
    assert_raises(TypeError, Ensemble, atoms, coords, [None, 'a'])


def test_ensemble_from_files():
    """Test Fragment from_file method."""
    dic = os.getenv('FDETADATA')
    xyz = 'he_test_'
    pqr = 'test_'
    ens = Ensemble.from_files(dic, pqr, extension='pqr')
    assert (ens.fragments[0].atoms == ['H','C', 'H']).all()
    ref_coords = np.array([[25.512, 38.671, 49.092],
                           [26.263, 39.012, 49.834],
                           [25.781, 39.262, 50.817]])
    assert np.allclose(ref_coords, ens.fragments[0].coords)
    ref_charges = np.array([0.074, 0.028, 0.057])
    assert np.allclose(ref_charges, ens.fragments[0].charges)
    ens = Ensemble.from_files(dic, xyz, extension='xyz')
    assert (ens.fragments[0].atoms == ['He', 'He']).all()
    assert np.allclose([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]], ens.fragments[0].coords)
    assert ens.fragments[0].charges is None


def test_common_members():
    """Test find common members between two lists."""
    a = [0, 1, 2, 3]
    b = [1, 3, 4, 5]
    c = [5, 6, 7, 8]
    assert common_members(a, b) == [1, 3]
    assert common_members(b, c) == [5]
    assert common_members(a, c) == []


def test_get_bond_matrix():
    """Test the bond matrix."""
    elements = ['H', 'H', 'He', 'O', 'H', 'H']
    geos = np.array([[10.0000, 0.000000, 0.000000],
                    [10.70400, 0.000000, 0.000000],
                    [0.000000, 0.000000, 0.000000],
                    [0.758602, 0.000000, 0.504284],
                    [0.758602, 0.000000, -0.504284]])
    geos = geos/BOHR
    assert_raises(ValueError, get_bond_matrix, elements, geos)
    elements = ['H', 'H', 'O', 'H', 'H']
    assert_raises(ValueError, get_bond_matrix, elements, geos, unit='vah')
    bond_matrix = get_bond_matrix(elements, geos)
    ref = np.array([[0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 1],
                    [0, 0, 1, 1, 0]])
    assert np.allclose(ref, bond_matrix)


def test_find_fragments():
    """Test finding fragments"""
    elements = ['H', 'H', 'O', 'H', 'H']
    geos = np.array([[10.0000, 0.000000, 0.000000],
                    [10.70400, 0.000000, 0.000000],
                    [0.000000, 0.000000, 0.000000],
                    [0.758602, 0.000000, 0.504284],
                    [0.758602, 0.000000, -0.504284]])
    geos = geos/BOHR
    ref = [[0, 1], [2, 3, 4]]
    assert_raises(ValueError, find_fragments, elements, geos, unit='vah')
    frag_list, frags = find_fragments(elements, geos)
    assert frag_list == ref
    assert np.allclose(frags[0], geos[ref[0]])
    assert np.allclose(frags[1], geos[ref[1]])


def test_get_interfragment_distances():
    """Test interfragment distance list"""
    elements = ['H', 'H', 'O', 'H', 'H']
    geos = np.array([[10.0000, 0.000000, 0.000000],
                    [10.70400, 0.000000, 0.000000],
                    [0.000000, 0.000000, 0.000000],
                    [0.758602, 0.000000, 0.504284],
                    [0.758602, 0.000000, -0.504284]])
    geos = geos/BOHR
    frag_list, frags = find_fragments(elements, geos)
    ds = get_interfragment_distances(frags[0], frags[1])
    assert len(ds) == 6
    assert ds[0] == 10.0000/BOHR
    assert ds[3] == 10.70400/BOHR


if __name__ == "__main__":
    test_align_basics()
    test_align_frames()
    test_fragment()
    test_fragment_from_file()
    test_ensemble()
    test_ensemble_from_files()
    test_common_members()
    test_get_bond_matrix()
    test_find_fragments()
    test_get_interfragment_distances()
