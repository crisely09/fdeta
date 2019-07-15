"""
Unit and regression test for the fdeta package.

"""

# Import package, test suite, and other packages as needed
import os
import numpy as np

from nose.tools import assert_raises
from fdeta.analysis import TrajectoryAnalysis
from fdeta.fdetmd.mdinterface import MDInterface


def test_mdinterface_base():
    """Test to initialize `MDInterface`."""
    # Define variables to break code
    dic = '/home/cris/code/fdemd_pub/fdeta/fdeta/'
    traj = os.path.join(dic, 'data/test_traj.xyz')
    box_size = np.array([4, 4, 4])
    grid_size = np.array([10, 10, 10])
    ta = TrajectoryAnalysis(traj)
    mdi = MDInterface(ta, box_size, grid_size)
    mdi.save_grid()
    ref_edges = [np.array([-1.8, -1.4, -1., -0.6, -0.2, 0.2, 0.6, 1., 1.4, 1.8]),
                 np.array([-1.8, -1.4, -1., -0.6, -0.2, 0.2, 0.6, 1., 1.4, 1.8]),
                 np.array([-1.8, -1.4, -1., -0.6, -0.2, 0.2, 0.6, 1., 1.4, 1.8])]
    guvref = np.loadtxt(os.path.join(dic, 'data/test_guv.txt'))
    guvhere = np.loadtxt(os.path.join(dic, 'tests/box_grid.txt'))
    np.allclose(mdi.edges, ref_edges)
    np.allclose(guvref[:,:-1], guvhere)
