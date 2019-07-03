"""
Tools for generating and modifying grids.

Creates, modifies, reads and writes different types of grids.
"""

import numpy as np
import scipy as sp


def simplify_cubic_grid(cubic_grid, var_values, tot_frames):
    """Simplify arrays from cubic grids to xyz type.

    Convert  and (N, N, N) arrays
    to a single (3 x N x N x N, 4).


    Parameters
    ----------
    cubic_grid : np.ndarray(dtype=float)
        Cubic grid from histogram
    var_values : np.ndarray(dtype=float)
        Values of the property at the `cubic_grid` points
    tot_frames : int
        Total number of MD frames used

    Returns
    -------
    output : np.ndarray(dtype=float), default=None
        Final array with grid points and values [x, y, z, value]

    """
    # Check variables
    if not isinstance(cubic_grid, np.ndarray):
        raise TypeError('cubic_grid must be numpy array')
    if not isinstance(var_values, np.ndarray):
        raise TypeError('var_values must be numpy array')
    if not isinstance(tot_frames, int):
        raise TypeError('tot_frames must be int')
    Nx, Ny, Nz = var_values.shape[0], var_values.shape[1], var_values.shape[2]
    if not Nx == Ny and not Nx == Nz:
        raise NotImplementedError("Only arrays with cubic forms accepted.")
    if cubic_grid.shape[1] < Nx:
        raise ValueError('cubic_grid has wrong shape.')

    # This expression does not work when Nx!=Ny or Nx!=Nz....
    delta = sp.diff(cubic_grid)
    cubic_grid = cubic_grid[:, :-1] + delta/2
    # Normalize
    var_values /= tot_frames
    # Divide on an elementary volume (of microcell)
    var_values /= delta[0][0] * delta[1][0] * delta[2][0]

    output = np.zeros((Nx*Ny*Nz, 4), dtype=float)
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                cnt = i + j*Ny + k*Ny*Nz
                # Loop over x
                output[cnt, 0] = cubic_grid[0][i]
                # Loop over y
                output[cnt, 1] = cubic_grid[1][j]
                # Loop over z
                output[cnt, 2] = cubic_grid[2][k]
                # Loop over PCF value (check consistency!)
                output[cnt, 3] = var_values[i][j][k]
    return output
