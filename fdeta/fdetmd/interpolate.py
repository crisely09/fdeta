"""

The Pair Correlation Function related tools.

"""

import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from fdeta.fdetmd.cgrid_tools import BoxGrid


def interpolate_function(refgrid: np.ndarray,
                         values: np.ndarray,
                         grid: np.ndarray, 
                         function: str='gaussian'):
    """ Interpolate some function to an external grid.

    This method assumes that the reference values are
    evaluated on the class' box grid.

    Parameters
    ----------
    refgrid : np.ndarray((n,3), dtype=float)
        Set of points where function was evaluated.
    grid : np.ndarray((n, 3), dtype=float)
        Grid where the interpolant will be evalated.
    function : string
        Name of method for the interpolation. Options are:
        `linear`, `cubic`, `gaussian`.

    Returns
    -------
    grid : np.ndarray
        Grid that contains coordinates and interpolated values.
    """
    # Prepare arrays for interpolator
    xs = refgrid[:, 0]
    ys = refgrid[:, 1]
    zs = refgrid[:, 2]
    interpolator = interpolate.Rbf(xs, ys, zs, values, function=function)
    # Replace values previously stored
    grid[:, 3] = interpolator(grid[:, 0], grid[:, 1], grid[:, 2])
    return grid


def interpolate_function_split(refgrid: np.ndarray,
                               values: np.ndarray,
                               grid: np.ndarray, 
                               function: str='gaussian',
                               max_memory: int = 2000):
    """ Interpolate some function to an external grid.

    This function splits the final grid in batches
    to reduce the memory needed for interpolation.

    Parameters
    ----------
    refgrid : np.ndarray((n,3), dtype=float)
        Set of points where function was evaluated.
    grid : np.ndarray((n, 3), dtype=float)
        Grid where the interpolant will be evalated.
    function : string
        Name of method for the interpolation. Options are:
        `linear`, `cubic`, `gaussian`.
    max_memory :  int
        The maximum size of cache to use (in MB)
    Returns
    -------
    grid : np.ndarray
        Grid that contains coordinates and interpolated values.
    """
    # Prepare arrays for interpolator
    xs = refgrid[:, 0]
    ys = refgrid[:, 1]
    zs = refgrid[:, 2]
    interpolator = interpolate.Rbf(xs, ys, zs, values, function=function)
    # Now split the grid into blocks to be able to use the interpolator
    # Define an appropriate size for the blocks
    BLKSIZE = 128
    npoints = len(grid)
    ngrids = grid.shape[0]
    blksize = int(max_memory*1e6/(npoints*8*BLKSIZE))*BLKSIZE
    blksize = max(BLKSIZE, min(blksize, ngrids, BLKSIZE*1200))
    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ngrids, ip0+blksize)
        coords = grid[ip0:ip1]
        grid[ip0:ip1, 3] = interpolator(coords[:, 0], coords[:, 1], coords[:, 2])
    return grid


def interpn(*args, **kw):
    """Interpolation on N-D. 

    ai = interpn(x, y, z, ..., a, xi, yi, zi, ...)
    Where the arrays x, y, z, ... define a rectangular grid
    and a.shape == (len(x), len(y), len(z), ...)
    Taken from: https://github.com/scipy/scipy/issues/2246

    """
    method = kw.pop('method', 'cubic')
    if kw:
        raise ValueError("Unknown arguments: " % kw.keys())
    nd = (len(args)-1)//2
    if len(args) != 2*nd+1:
        raise ValueError("Wrong number of arguments")
    q = args[:nd]
    qi = args[nd+1:]
    a = args[nd]
    for j in range(nd):
        a = interp1d(q[j], a, axis=j, kind=method)(qi[j])
    return a
