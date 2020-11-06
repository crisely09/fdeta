"""
Unit and regression test for the fdeta package.

"""

# Import package, test suite, and other packages as needed
import os
import numpy as np
from nose.tools import assert_raises

from fdeta.radial_distributions import centroid_distance, center_of_mass_distance
from fdeta.radial_distributions import compute_rad
from fdeta.units import BOHR


def test_compute_rad():
    """Test the radial average distribution."""
    points = np.array([[0.758602, 0.000000, 0.504284],
                       [0.758602, 0.000000, -0.504284]])
    values = np.zeros((10, 4))
    for i in range(10):
        values[i, 2] = 0.05 + i*.1
        values[i, 1] = i
        values[i, 3] = 0.5*i
    xs, ys = compute_rad(points, values, bins=5, limits=None)
    

if __name__ == "__main__":
    test_compute_rad()
