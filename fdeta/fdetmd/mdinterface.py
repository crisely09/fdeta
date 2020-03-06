"""

The Pair Correlation Function related tools.

"""

import numpy as np
import scipy as sp
from typing import Union
from fdeta.fdetmd.cgrid_tools import BoxGrid
from fdeta.mdtrajectory import MDTrajectory
from fdeta.fdetmd.interpolate import interpolate_function
from fdeta.cube import write_cube


def check_grid(grid):
    """Check if the grid is from file or it is already an array."""
    if isinstance(grid, str):
        grid = np.loadtxt(grid)
    elif not isinstance(grid, np.ndarray):
        raise TypeError("Final grid `fingrid` must be str or np.nparray")
    return grid


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
        # Save aligned fragment to file
        self.ta_object.align_along_trajectory(frag_id, to_file=True)
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
        extgrid = self.interpolate(rhob[:, :3], rhob[:, 3], gridname)
        return extgrid

    @staticmethod
    def interpolate(refgrid: np.ndarray, values: np.ndarray,
                    fingrid: Union[str, np.ndarray], function: str = 'gaussian'):
        """ Interpolate some function to an external grid.

        This method assumes that the reference values are
        evaluated on the class' box grid.

        Parameters
        ----------
        refgrid : np.ndarray((n,3), dtype=float)
            Set of points where function was evaluated.
        fingrid : string or numpy ndarray
            File with new set of points for the interpolation.
        function : string
            Name of method for the interpolation. Options are:
            `linear`, `cubic`, `gaussian`.

        Returns
        -------
        interpolated : np.ndarray((n, 4)), dtype=float)
            Array with gridpoints and interpolated values.

        """
        fingrid = check_grid(fingrid)
        interpolated = interpolate_function(refgrid, values, fingrid, function)
        return interpolated

    def compute_electrostatic_potential(self, charge_coeffs: dict,
                                       fingrid: Union[str, np.ndarray],
                                       fname: str = 'elst_pot.txt'):
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

        fingrid = check_grid(fingrid)
        # Clean the weights from grid to leave space for the potential
        fingrid[:, 3] = 0.0
        self.pbox.electrostatic_potential(self.npoints, self.total_frames,
                                          fname, charge_density, fingrid)

    def export_cubefile(self, frag_id, geometry: Union[int, np.ndarray],
                        values: np.ndarray, is_sorted: bool = False,
                        fname: str = "md_interface.cube"):
        """ Create cubefile from data and internal BoxGrid grid.

        Parameters
        ----------
        frag_id : int
            ID of fragment to use. Defines which atoms to use.
        geometry: Optional[int, np.ndarray]
            If int given, the curresponding Frame is taken from the trajectory.
            If array given, expecting [x, y, z] coordinates in Angstrom.
        fname : str
            Name for cubefile.

        Exports
        -------
        Cubefile in Gaussian format.
        """
        atoms = self.ta_object.trajectory[frag_id]['elements']
        if isinstance(geometry, int):
            coords = self.ta_object.trajectory[frag_id]['geometries'][geometry]/self.bohr
        elif isinstance(geometry, np.ndarray):
            coords = geometry/self.bohr
        else:
            raise ValueError("""Geometry must be given as in int to indicate the frame number"""
                             """or as a np.ndarray (n, 3) with xyz coordinates in Angstrom.""")
        # Make objects to build cubic grid
        grid = np.zeros((len(values), 3))
        grid = self.pbox.get_grid(corder=False)
        step = grid[1, 0] - grid[0, 0]
        vectors = np.zeros((3, 3))
        origin = np.zeros((3))
        for i in range(3):
            vectors[i, i] = step
            origin[i] = grid[0, i]
        if is_sorted is False:
            values = self.sort_values_cube(self.box_size, values)
        if '.cube' not in fname:
            fname = fname+'.cube'
        write_cube(atoms, coords, origin, vectors, self.box_size, values,
                   fname)

    @staticmethod
    def sort_values_cube(box_size: tuple, ref_values: np.ndarray):
        """Sort values to follow cubefile order.

        Numpy mesh grids and histograms use the inverse order to cubefiles,
        the fastest variable is x, and the slowest is z.

        Parameters
        ----------
        box_size : tuple
            Amount of points in each coordinate.
        ref_values : np.ndarray
            Values in reference (numpy) order.

        Returns
        -------
        values :  np.ndarray
            Re-arranged values, sorted so they match with the cubefile order,
            z being the fastest variable and x the slowest.
        """
        nx = box_size[0]
        ny = box_size[1]
        nz = box_size[2]
        icount = 0
        values = np.zeros(ref_values.shape)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    fcount = x + y*ny + z*ny*nz
                    values[icount] = ref_values[fcount]
                    icount += 1
        return values
