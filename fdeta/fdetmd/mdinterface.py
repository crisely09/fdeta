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
        self.bohr = 0.529177249

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

    def get_elec_density(self, charge_coeffs, ingrid=False):
        """ Evaluate the electronic density of the solvent.

        Parameters
        ----------
        charge_coeffs : dict('element' : coeff)
            The ratio between effective charge and nuclear charge: q_B/Z_B.

        Returns
        -------
        rhob : np.ndarray((npoints, 4), dtype=float)
            Density of solvent on npoints, everything in a.u.
            ingrid must be set True
        rhocube : np.ndarray(npoints, dtype=float)
            Density of electrons of solvent in cubic grid, in Angstrom.

        """
        rhocube = None
        for ielement in list(charge_coeffs.keys()):
            print("ielement: ", ielement)
            if rhocube is None:
                print(self.pcf[ielement])
                rhocube = (-charge_coeffs[ielement]*self.ta_object.nametocharge(ielement)
                           * self.pcf[ielement])
            else:
                print(self.pcf[ielement])
                rhocube -= (charge_coeffs[ielement]*self.ta_object.nametocharge(ielement)
                            * self.pcf[ielement])
        if ingrid:
            rhob = self.pbox.normalize(self.npoints*4, self.total_frames, rhocube)
            rhob = np.reshape(rhob, (self.npoints, 4))
            dv = self.delta[0][0] * self.delta[1][0] * self.delta[2][0]
            print("dv = %.12f" % dv)
            rhob[:, 3] /= dv
            rhob[:, 3] *= self.bohr**3
            return rhob
        else:
            return rhocube

    def get_nuclear_density(self, ingrid=False):
        """ Evaluate the nuclear charge density of the solvent.

        Parameters
        ----------
        ingrid : bool
            If nuclear charges must be returned with the grid or not.

        Returns
        -------
        nuc_charges : np.ndarray((npoints, 4), dtype=float)
            Density of solvent on npoints, everything in a.u.
            ingrid must be set True
        nuclei : np.ndarray(npoints, dtype=float)
            Density charge of solvent in cubic grid, in Angstrom.

        """
        nuclei = None
        for ielement in list(self.pcf.keys()):
            if nuclei is None:
                nuclei = self.ta_object.nametocharge(ielement)*self.pcf[ielement]
            else:
                nuclei += self.ta_object.nametocharge(ielement)*self.pcf[ielement]
        if ingrid:
            nuc_charges = self.pbox.normalize(self.npoints*4, self.total_frames, nuclei)
            nuc_charges = np.reshape(nuc_charges, (self.npoints, 4))
            dv = self.delta[0][0] * self.delta[1][0] * self.delta[2][0]
            nuc_charges[:, 3] /= dv
            print("bohr^3 = ", self.bohr**3)
            nuc_charges[:, 3] *= self.bohr**3
            return nuc_charges
        else:
            return nuclei

    def rhob_on_grid(self, charge_coeffs, gridname='extragrid.txt'):
        """ Evaluate rhoB on an specific grid.

        Parameters
        ----------
        extgrid : np.ndarray((n,3), dtype=float)
            New set of points for the interpolation.

        """
        rhob = self.get_elec_density(charge_coeffs, ingrid=True)
        # Normalize charge with respect to volume element
        rhob[:, 3] *= -1.0
        dv = self.delta[0][0] * self.delta[1][0] * self.delta[2][0] / self.bohr**3
        rhob[:, 3] /= dv
        np.savetxt('refrhob.txt', rhob)
        extgrid = self.interpolate_function(rhob[:, :3], rhob[:, 3], gridname)
        np.savetxt('rhob.txt', extgrid)

    @staticmethod
    def interpolate_function(refgrid, values, gridname='extragrid.txt', method='linear'):
        """ Interpolate some function to an external grid.

        This method assumes that the reference values are
        evaluated on the class' box grid.

        Parameters
        ----------
        refgrid : np.ndarray((n,3), dtype=float)
            Set of points where function was evaluated.
        function : np.ndarray(N, dtype=float)
            Reference function values to create interpolator.
        gridname : string
            File with new set of points for the interpolation.
        method : string
            Name of method for the interpolation. Options are:
            `linear`, `cubic`.

        """
        from scipy import interpolate
        xi = refgrid[:, 0]
        yi = refgrid[:, 1]
        zi = refgrid[:, 2]
        interpolator = interpolate.Rbf(xi, yi, zi, values, function=method)
        # interpolator = interpolate.LinearNDInterpolator(rhob[: ,:3], rhob[:, 3])
        # Load grid and clean weights
        extgrid = np.loadtxt(gridname)
        extgrid[:, 3] = 0.0
        xj = extgrid[:, 0]
        yj = extgrid[:, 1]
        zj = extgrid[:, 2]
        extgrid[:, 3] = interpolator(xj, yj, zj)
        # extgrid[:, 3] = interpolator(extgrid[:, :3])
        return extgrid

    def compute_electrostatic_potential(self, charge_coeffs, gridname='extragrid.txt'):
        """ Evaluate and save electrostatic potential.

        Parameters
        ----------
        charge_coeffs : dict('element' : coeff)
            The ratio between effective charge and nuclear charge: q_B/Z_B.
        extgrid : np.ndarray((n,3), dtype=float)
            New set of points for the interpolation.
            This grid must be in Bohr!

        """
        net_density = self.get_elec_density(charge_coeffs)
        net_density += self.get_nuclear_density()
        charge_density = self.pbox.normalize(self.npoints*4, self.total_frames, net_density)
        extgrid = np.loadtxt(gridname)
        # Clean the weights from grid to leave space for the potential
        extgrid[:, 3] = 0.0
        oname = 'elects_pot.txt'
        self.pbox.electrostatic_potential(self.npoints, self.total_frames,
                                          oname, charge_density, extgrid)
