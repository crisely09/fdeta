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
    def __init__(self, ta_object, box_size, mol_id=0):
        """ Create a pcf ta_object.

        Parameters
        ----------
        ta_object : Class_Trajectory_Analysis
            Object with all information about the MD trajectory
        box_size : tuple(3)
            Size of cubic grid where to evaluate the PCF
        grid : np.ndarray((Npoints,3), dtype=float)
            Coordinates of cubic grid. This should be the same
            for all the atoms.
        values : list(np.ndarray(Npoints))
            List of histogram information for each atom, in 1D.
        mol_id : int
            Index indicating which molecule(s) to take as solute.
            By default solute = 0.

        """
        self.ta_object = ta_object
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
                                                                           box_size, mol_id)
        delta = sp.diff(edges)
        edges = np.array(edges)
        # TODO: make sure next line works for any size of histogram
        self.edges = edges[:, :-1] + delta/2.0
        self.total_frames = self.ta_object.Total_number_of_frames
        self.dvolume = delta[0][0] * delta[1][0] * delta[2][0]
        self.ta_object.select(self.solvent)
        self.solvent_atomtypes = self.ta_object.Topology[self.solvent][1]
        self.solvent_natoms = len(self.solvent_atomtypes)
        # Initialize Pybind Class
        self.pbox = BoxGrid(box_size, edges)

    @staticmethod
    def save_grid(box_size, step_size, pbox, fname='box_grid.txt'):
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
        steps = box_size/step_size
        npoints = np.cumprod(steps)[-1]
        grid = np.zeros((npoints, 3))
        # Call the cpp pybind11 implementation
        pbox.save_grid(grid, fname)

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
