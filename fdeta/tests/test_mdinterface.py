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
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.xyz')
    box_size = np.array([4, 4, 4])
    grid_size = np.array([10, 10, 10])
    ta = TrajectoryAnalysis(traj)
    mdi = MDInterface(ta, box_size, grid_size)
    mdi.save_grid()
    ref_edges = [np.array([-1.8, -1.4, -1., -0.6, -0.2, 0.2, 0.6, 1., 1.4, 1.8]),
                 np.array([-1.8, -1.4, -1., -0.6, -0.2, 0.2, 0.6, 1., 1.4, 1.8]),
                 np.array([-1.8, -1.4, -1., -0.6, -0.2, 0.2, 0.6, 1., 1.4, 1.8])]
    guvref = np.loadtxt(os.path.join(dic, 'test_guv.txt'))
    guvhere = np.loadtxt(os.path.join(dic, 'box_grid.txt'))
    np.allclose(mdi.points, ref_edges)
    np.allclose(guvref[:,:-1], guvhere)
test_mdinterface_base()

def test_mdinterface_histogram():
    """Test to initialize `MDInterface`."""
    # Define variables to break code
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.xyz')
    box_size = np.array([10, 10, 10])
    grid_size = np.array([10, 10, 10])
    ta = TrajectoryAnalysis(traj)
    mdi = MDInterface(ta, box_size, grid_size)
    mdi.save_grid('second_text.txt')
    assert (mdi.delta == 1.0).all()
    ccoeffs = {'O': 1.1, 'H': 0.6}
    rho = mdi.get_elec_density(ccoeffs)
    assert np.sum(rho)/2 == -20
    gridname = os.path.join(dic, 'grid_vemb.dat')
    mdi.compute_electrostatic_potential(ccoeffs, gridname)
    mdi.rhob_on_grid(ccoeffs, gridname)
