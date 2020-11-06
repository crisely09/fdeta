"""
Unit and regression test for the fdeta package.

"""

# Import package, test suite, and other packages as needed
import os
import numpy as np
from nose.tools import assert_raises

from fdeta.fragments import get_bond_matrix, common_members
from fdeta.fragments import find_fragments, get_interfragment_distances
from fdeta.units import BOHR


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
    test_common_members()
    test_get_bond_matrix()
    test_find_fragments()
    test_get_interfragment_distances()
