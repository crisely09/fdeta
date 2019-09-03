"""
Unit and regression test for the fdeta package.

"""

# Import package, test suite, and other packages as needed
import os
import numpy as np

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
    np.allclose(guvref[:, :-1], guvhere)


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


def test_mdinterface_acetone_w2():
    """Test  `MDInterface`."""
    # Define variables
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'traj_acetone_w2.xyz')
    box_size = np.array([10, 10, 10])
    grid_size = np.array([10, 10, 10])
    ta = TrajectoryAnalysis(traj)
    mdi = MDInterface(ta, box_size, grid_size)
    mdi.save_grid()
    gridname = os.path.join(dic, 'grid_vemb_acetone.dat')
    ccoeffs = {'O': 1.1, 'H': 0.6}
    rho = mdi.get_elec_density(ccoeffs, ingrid=True)
    rho[:, :3] *= mdi.bohr
    np.savetxt('rhob_acetone.txt', rho)
    nuc = mdi.get_nuclear_density(ingrid=True)
    np.savetxt('nuc_charge_acetone.txt', nuc)
    total = rho[:, 3] + nuc[:, 3]
    np.savetxt('total_charge.txt', total)
    mdi.compute_electrostatic_potential(ccoeffs, gridname)


if __name__ == "__main__":
    test_mdinterface_base()
    test_mdinterface_histogram()
    test_mdinterface_acetone_w2()
