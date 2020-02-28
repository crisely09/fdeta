"""

The Pair Correlation Function related tools.

"""

import numpy as np
import scipy as sp
from typing import Union
from fdeta.fdetmd.cgrid_tools import BoxGrid
from fdeta.mdtrajectory import MDTrajectory
from fdeta.fdetmd.interpolate import interpolate_function


class MDInterface:
    """Interface between MD trajectories and FDET elements.

    Attributes
    ----------
    ta_object : TrajectoryAnalysis object

    """
    def __init__(self, ta_object: MDTrajectory,
                 box_size: tuple, grid_size: np.ndarray,
                 frag_id: int = 0, average_solute: bool = False):
        """ Create a pcf ta_object.

        Parameters
        ----------
        ta_object : MDTrajectory
            Object with all information about the MD trajectory
        box_size : tuple(3)
            Size of cubic grid where to evaluate the PCF
        grid_size : np.ndarray((Npoints,3), dtype=float)
            Number of points in each direction.
        frag_id : int
            Index indicating which molecule(s) to take as solute.
            By default solute = 0.
        average_solute : bool
            Whether the solute molecules also need to be averaged.

        """
        self.ta_object = ta_object
        histogram_range = np.asarray([-box_size/2., box_size/2.]).T
        self.ta_object.align_along_trajectory(frag_id, self.ta_object.topology, to_file=True)
        if average_solute:
            self.ta_object.get_average_structure(frag_id)
        # Align solvent and find the averaged structure
        edges, self.pcf = self.ta_object.compute_pair_correlation_function(histogram_range,
                                                                           grid_size, frag_id)
        self.grid_size = grid_size
        self.npoints = np.cumprod(grid_size)[-1]
        self.delta = sp.diff(edges)
        edges = np.array(edges)
        # NOTE: only works for cubic grids
        self.points = edges[:, :-1] + self.delta/2.0
        self.total_frames = self.ta_object.nframes
        # Initialize Pybind Class
        self.pbox = BoxGrid(grid_size, self.points)
        self.bohr = 0.529177249

    def save_grid(self, fname: str = 'box_grid.txt'):
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

    def pcf_from_file(self, filename: str, element: Union[str, int]):
        """ Read the pcf from file."""
        raise NotImplementedError

    def pcf_to_file(self, filename: str, element: Union[str, int]):
        """ Save the pcf to file."""
        raise NotImplementedError

    def get_elec_density(self, charge_coeffs: dict, ingrid: bool = False):
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
        for ielement in charge_coeffs:
            if rhocube is None:
                rhocube = (-charge_coeffs[ielement]*self.ta_object.charges[ielement][1]
                           * self.pcf[ielement])
            else:
                rhocube -= (charge_coeffs[ielement]*self.ta_object.charges[ielement][1]
                            * self.pcf[ielement])
        if ingrid:
            rhob = self.pbox.normalize(self.npoints*4, self.total_frames, rhocube, False)
            rhob = np.reshape(rhob, (self.npoints, 4))
            dv = self.delta[0][0] * self.delta[1][0] * self.delta[2][0]
            rhob[:, 3] /= dv
            rhob[:, 3] *= self.bohr**3
            return rhob
        else:
            return rhocube

    def get_nuclear_density(self, ingrid: bool = False):
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
        for ielement in self.pcf:
            if nuclei is None:
                nuclei = self.ta_object.charges[ielement][1]*self.pcf[ielement]
            else:
                nuclei += self.ta_object.charges[ielement][1]*self.pcf[ielement]
        if ingrid:
            nuc_charges = self.pbox.normalize(self.npoints*4, self.total_frames,
                                              nuclei, False)
            nuc_charges = np.reshape(nuc_charges, (self.npoints, 4))
            dv = self.delta[0][0] * self.delta[1][0] * self.delta[2][0]
            nuc_charges[:, 3] /= dv
            nuc_charges[:, 3] *= self.bohr**3
            return nuc_charges
        else:
            return nuclei

    def get_rhob(self, charge_coeffs: dict, gridname: str = 'extragrid.txt'):
        """ Evaluate rhoB on an specific grid.

        Parameters
        ----------
        charge_coeffs : dict('element' : coeff)
            The ratio between effective charge and nuclear charge: q_B/Z_B.
        gridname : str
            New set of points for the interpolation.

        """
        rhob = self.get_elec_density(charge_coeffs, ingrid=True)
        # Normalize charge with respect to volume element
        rhob[:, 3] *= -1.0
        # np.savetxt('refrhob.txt', rhob)
        print("rho shape", rhob.shape)
        extgrid = self.interpolate(rhob[:, :3], rhob[:, 3], gridname)
        return extgrid

    @staticmethod
    def interpolate(refgrid: np.ndarray, values: np.ndarray,
                    gridname: str, function='gaussian'):
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
        function : string
            Name of method for the interpolation. Options are:
            `linear`, `cubic`, `gaussian`.

        """
        grid = np.loadtxt(gridname)
        extgrid = interpolate_function(refgrid, values, grid, function)
        return extgrid

    def compute_electrostatic_potential(self, charge_coeffs: dict,
                                        gridname: str = 'extragrid.txt'):
        """ Evaluate and save electrostatic potential.

        Parameters
        ----------
        gridname : str
            New set of points for the interpolation.
            This grid must be in Bohr!

        """
        net_density = self.get_elec_density(charge_coeffs)
        net_density += self.get_nuclear_density()
        charge_density = self.pbox.normalize(self.npoints*4, self.total_frames, net_density,
                                             False)
        extgrid = np.loadtxt(gridname)
        # Clean the weights from grid to leave space for the potential
        extgrid[:, 3] = 0.0
        oname = 'elects_pot.txt'
        self.pbox.electrostatic_potential(self.npoints, self.total_frames,
                                          oname, charge_density, extgrid)

    def export_cubefile(self, atoms: Union[list, np.ndarray],
                        coords: np.ndarray, grid_values: np.ndarray,
                        only_values: bool = False):
        """ Create cubefile from data and grid.
        """
        if only_values:

        # Make objects to build cubic grid
        step = grid_values[1, 0] - grid_values[0, 0]
        vectors = np.zeros((3, 3))
        origin = np.zeros((3))
        values = np.zeros(grid_values.shape)
        for i in range(3):
            vectors[i, i] = step
            origin[i] = rho[0, i]
        # Re-order values to right cubefile format
        nx = self.grid_size[0]
        ny = self.grid_size[1]
        nz = self.grid_size[2]
        icount = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    fcount = x + y*ny + z*ny*nz
                    values[icount] = rho[fcount, 3]
                    icount += 1
        # cube_grid = make_grid(grid_shape, vectors, origin)
        # np.savetxt("cubic_grid.txt", cube_grid)

