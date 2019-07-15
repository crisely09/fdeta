"""

The Pair Correlation Function related tools.

"""

import numpy as np
import scipy as sp
from fdeta.fdetmd.cgrid_tools import BoxGrid


class MDInterface(object):
    """Interface between MD trajectories and FDET elements.

    Attributes
    ----------
    ta_object : TrajectoryAnalysis object

    """
    def __init__(self, ta_object, box_size, grid_size, mol_id=0):
        """ Create a pcf ta_object.

        Parameters
        ----------
        ta_object : Class_Trajectory_Analysis
            Object with all information about the MD trajectory
        box_size : tuple(3)
            Size of cubic grid where to evaluate the PCF
        grid_size : np.ndarray((Npoints,3), dtype=float)
            Number of points in each direction.
        mol_id : int
            Index indicating which molecule(s) to take as solute.
            By default solute = 0.

        """
        self.ta_object = ta_object
        self.box_size = box_size
        self.grid_size = grid_size
        if mol_id == 0:
            self.solute = 0
            self.solvent = 1
        else:
            self.solute = 1
            self.solvent = 0
        # TODO: verify the range is consistent with any step size.
        histogram_range = np.asarray([-box_size/2., box_size/2.]).T
        self.ta_object.select(mol_id)
        self.ta_object.align_along_trajectory(mol_id, self.ta_object.Topology)
        edges, self.pcf = self.ta_object.compute_pair_correlation_function(histogram_range,
                                                                           grid_size, mol_id)
        self.delta = sp.diff(edges)
        edges = np.array(edges)
        # TODO: make sure next line works for any size of histogram
        self.edges = edges[:, :-1] + self.delta/2.0
        self.total_frames = self.ta_object.Total_number_of_frames
        self.dvolume = self.delta[0][0] * self.delta[1][0] * self.delta[2][0]
        self.ta_object.select(self.solvent)
        self.solvent_atomtypes = self.ta_object.Topology[self.solvent][1]
        self.solvent_natoms = len(self.solvent_atomtypes)
        # Initialize Pybind Class
        self.pbox = BoxGrid(grid_size, self.edges)

    def save_grid(self, fname='box_grid.txt'):
        """ Get xyz grid into text file.

        Parameters
        ----------
        box_size : tuple(3), int
            Size of grid box.
        edges : list(np.ndarray(3, dtype=float))
            Bin edges of each dimension.
        step_size : float
            Step size in Angstrom

        """
        steps = self.grid_size/self.delta[:, 0]
        npoints = int(np.cumprod(steps)[-1])
        grid = np.zeros((npoints, 3))
        # Call the cpp pybind11 implementation
        self.pbox.save_grid(grid, fname)

    def pcf_from_file(self, filename, element):
        """ Read the pcf from file."""
        raise NotImplementedError

    def pcf_to_file(self, filename, element):
        """ Save the pcf to file."""
        raise NotImplementedError

    def save_rhob_ongrid(self, grid):
        """ Evaluate rhoB on an specific grid."""
        raise NotImplementedError

    def interpolate_function(self, function, extgrid):
        """ Interpolate some function to an external grid.

        This method assumes that the reference values are
        evaluated on the class' box grid.

        Parameters
        ----------
        function : np.ndarray(N, dtype=float)
            Reference values to create interpolator.
        extgrid : np.ndarray((n,3), dtype=float)
            New set of points for the interpolation.

        """
        raise NotImplementedError

    def compute_electrostatic_potential(self, extgrid, weights, density):
        """ Evaluate and save electrostatic potential.

        Parameters
        ----------
        extgrid :

        """
        raise NotImplementedError
