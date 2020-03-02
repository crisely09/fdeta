"""
Unit and regression test for the fdeta package.

"""

# Import package, test suite, and other packages as needed
import os
import numpy as np

from fdeta.fdetmd.auxfns import integrate


def test_integrate():
    """Test to initialize `MDInterface."""
    # Define variables to break code
    dic = os.getenv('FDETADATA')
    fname1 = os.path.join(dic, 'grid_vemb_acetone.dat')
    fname2 = os.path.join(dic, 'grid_rhoA_acetone.dat')
    weights = np.loadtxt(fname1)
    densA = np.loadtxt(fname2)
    ref = np.dot(densA[:, 3], weights[:, 3])
    elec = integrate(weights[:, 3], densA[:, 3])
    assert abs(ref - elec) < 1e-9


if __name__ == "__main__":
    test_integrate()
