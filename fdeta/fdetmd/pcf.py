"""

The Pair Correlation Function related tools.

"""

import numpy as np
import scipy as sp


class PCFClass(object):
    """PCF object."""
    def __init__(self, ta_object, box_size, grid, values, mol_id=0):
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
        self.grid = grid
        self.values = values
        if mol_id == 0:
            self.solute = 0
            self.solvent = 1
        else:
            self.solute = 1
            self.solvent = 0
        ta_object.select(self.solvent)
        self.solvent_atomtypes = ta_object.Topology[self.solvent][1]
        self.solvent_natoms = len(self.solvent_atomtypes)

    def from_trajectory(self, ta_object, box_grid, box_size, mol_id):
        """ Create a pcf from trajectory.

        Parameters
        ----------

        """
        histogram_range = np.asarray([-box_size/2., box_size/2.]).T
        pcf_fromtraj = ta_object.compute_pair_correlation_function(histogram_range,
                                                                   box_grid, mol_id)
        edges = pcf_fromtraj[0][1]  # We take the first element, but it could be any
        delta = sp.diff(edges)
        edges = edges[:, :-1] + delta/2.0
        # print delta
        grid = self.boxgrid_to_stdgrid(box_size, edges)
        total_frames = ta_object.Total_number_of_frames
        elementary_volume = delta[0][0] * delta[1][0] * delta[2][0]
        values = self.linearize_histograms(pcf_fromtraj, box_size, total_frames,
                                           elementary_volume)
        self.__init__(ta_object, grid, box_size, values, mol_id)

    @staticmethod
    def boxgrid_to_stdgrid(box_size, edges):
        steps = box_size*2  # Steps are of 0.5A
        npoints = np.cumprod(steps)[-1]
        grid = np.zeros((npoints, 3))
        # Call the cpp pybind11 implementation
        from cgrid_tools import box_to_std
        box_to_std(steps, edges, grid)
        return grid

    @staticmethod
    def linearize_histograms(pcf_fromtraj, box_size, total_frames, elementary_volume):
        """Convert values from histogram into a 1D array."""
        # Normalize with the nomber of frames
        # Divide by elementary volume
        from cgrid_tools import make_hist1d
        natoms = len(pcf_fromtraj)
        values = []
        for i in range(natoms):
            hist = pcf_fromtraj[i][0]
            hist /= total_frames
            hist /= elementary_volume
            val = make_hist1d(box_size, hist)
            values.append(val)
        return values

    def from_file(filename):
        """ Read the pcf from file."""
        raise NotImplementedError

    def to_file(filename, split=False):
        """ Save the pcf to file."""
        raise NotImplementedError
