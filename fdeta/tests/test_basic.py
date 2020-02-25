
# Import package, test suite, and other packages as needed
import os
import numpy as np

from fdeta.analysis import TrajectoryAnalysis
from fdeta.mdtrajectory import MDTrajectory
from fdeta.fdetmd.mdinterface import MDInterface
from fdeta.fdetmd.dft import compute_nad_lda_all

def test():
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.xyz')
    box_size = np.array([4, 4, 4])
    grid_size = np.array([10, 10, 10])
    histogram_range = np.asarray([-box_size/2., box_size/2.]).T
    ta = TrajectoryAnalysis(traj)
    ta.select(0)
    ta.align_along_trajectory(0, ta.Topology)
    ta.select(1)
    ta.align_along_trajectory(1, ta.Topology)
    ta.get_average_structure(1)
    edges, pcf = ta.compute_pair_correlation_function(histogram_range,                                                                      grid_size, 0)
   #self.npoints = np.cumprod(grid_size)[-1]
   #self.delta = sp.diff(edges)
   #edges = np.array(edges)
   ## NOTE: only works for cubic grids
   #self.points = edges[:, :-1] + self.delta/2.0
   #self.total_frames = self.ta_object.Total_number_of_frames


def test_base():
    """Basic checks for MDTrajectory class."""
    dic = os.getenv('FDETADATA')
    ftraj = 'he_traj.txt'
    fcharges = 'he_charges.txt'
    traj = MDTrajectory(ftraj, fcharges) 
    structure = traj.get_structure_from_topology(0, 0, traj.topology)
    assert np.allclose(structure, [0., 0., 0.])
    traj.save_topology(0)
    with open('topology_0.txt', 'r') as tf:
        text = tf.read()
    reftext = """1\nFrame 0\nHe 0.0 0.0 0.0 0\n"""
    assert text == reftext 
    os.remove('topology_0.txt')

def test_pcf():
    """Test if pcf works"""
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'test_traj.xyz')
    ta = MDTrajectory(traj)
    box_size = np.array([4, 4, 4])
    grid_size = np.array([10, 10, 10])
    histogram_range = np.asarray([-box_size/2., box_size/2.]).T
    edges, pcf = ta.compute_pair_correlation_function(histogram_range,                                                                      grid_size, 0)
    pcfO = np.where(pcf["O"] == 1.0)
    assert np.allclose(pcfO, (np.array([1, 8]), np.array([4, 6]), np.array([5, 4])))

if __name__ == "__main__":
    test()
    test_base()
    test_pcf()
