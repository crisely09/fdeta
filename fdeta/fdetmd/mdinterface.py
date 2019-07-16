"""

The Pair Correlation Function related tools.

"""

import numpy as np
import scipy as sp
from fdeta.fdetmd.cgrid_tools import BoxGrid


class MDInterface:
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
        histogram_range = np.asarray([-box_size/2., box_size/2.]).T
        self.ta_object.select(mol_id)
        self.ta_object.align_along_trajectory(mol_id, self.ta_object.Topology)
        self.ta_object.get_average_structure(mol_id)
        edges, self.pcf = self.ta_object.compute_pair_correlation_function(histogram_range,
                                                                           grid_size, mol_id)
        self.npoints = np.cumprod(grid_size)[-1]
        self.delta = sp.diff(edges)
        edges = np.array(edges)
        # NOTE: only works for cubic grids
        self.points = edges[:, :-1] + self.delta/2.0
        self.total_frames = self.ta_object.Total_number_of_frames
        # Initialize Pybind Class
        self.pbox = BoxGrid(grid_size, self.points)

    def save_grid(self, fname='box_grid.txt'):
        """ Get xyz grid into text file.

        Parameters
        ----------
        box_size : tuple(3), int
            Size of grid box.
        step_size : float
            Step size in Angstrom

        """
        # Call the cpp pybind11 implementation
        self.pbox.save_grid(self.npoints*3, fname)

    def pcf_from_file(self, filename, element):
        """ Read the pcf from file."""
        raise NotImplementedError

    def pcf_to_file(self, filename, element):
        """ Save the pcf to file."""
        raise NotImplementedError

    def get_rhob(self, charge_coeffs):
        """ Evaluate the density of the solvent (other fragment from mol_id).

        Parameters
        ----------
        charge_coeffs : dict('element' : coeff)
            The ratio between effective charge and nuclear charge: q_B/Z_B.

        Returns
        -------
        rhob : np.ndarray((npoints, 4), dtype=float)
            Density of solvent on npoints, everything in a.u.

        """
        rhocube = None
        for ielement in list(charge_coeffs.keys()):
            if rhocube is None:
                rhocube = (-charge_coeffs[ielement]*self.ta_object.nametomass(ielement)
                           *self.pcf[ielement])
            else:
                rhocube -= (charge_coeffs[ielement]*self.ta_object.nametomass(ielement)
                            *self.pcf[ielement])
        dvolume = self.delta[0][0] * self.delta[1][0] * self.delta[2][0]
        rhob = self.pbox.normalize(self.npoints*4, self.total_frames, dvolume, rhocube)
        rhob = np.reshape(rhob, (self.npoints, 4))
        return rhob

    def get_nuclear_charges(self):
        """ Evaluate the nuclear charges of the solvent (other fragment from mol_id).

        Returns
        -------
        nuc_charges : np.ndarray((npoints, 4), dtype=float)
            Density of solvent on npoints, everything in a.u.

        """
        nuclei = None
        for ielement in list(charge_coeffs.keys()):
            if nuclei is None:
                nuclei = self.ta_object.nametomass(ielement)*self.pcf[ielement]
            else:
                nuclei += self.ta_object.nametomass(ielement)*self.pcf[ielement]
        dvolume = self.delta[0][0] * self.delta[1][0] * self.delta[2][0]
        nuc_charges = self.pbox.normalize(self.npoints*4, self.total_frames, dvolume, nuclei)
        nuc_charges = np.reshape(nuc_charges, (self.npoints, 4))
        return nuc_charges

    def save_rhob_ongrid(self, extgrid=None):
        """ Evaluate rhoB on an specific grid.

        Parameters
        ----------
        extgrid : np.ndarray((n,3), dtype=float)
            New set of points for the interpolation.

        """
        if not extgrid:
            extgrid = np.loadtxt('extgrid.txt')
        grid = np.zeros((self.npoints, 3))
        grid = self.pbox.get_grid(grid)
        raise NotImplementedError

    @staticmethod
    def interpolate_function(function, extgrid):
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
