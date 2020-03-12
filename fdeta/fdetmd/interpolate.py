"""

The Pair Correlation Function related tools.

"""

import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from fdeta.fdetmd.cgrid_tools import BoxGrid
from fdeta.cube import make_cubic_grid


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


def interpolate_function_sectors(cube: dict,
                                 grid: np.ndarray, 
                                 npoints: int = 20,
                                 function: str='gaussian'):
    """ Interpolate some function to an external grid.

    This function splits the final grid in sectors of space
    to reduce the memory needed for interpolation.

    Parameters
    ----------
    cube : np.ndarray((n,3), dtype=float)
        Set of points where function was evaluated.
    ref_shape : tuple
        The shape of reference grid.
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
    # Find range of each grid
    # and common range to work with
    common_range = []
    splits = []
    cgrid_shape = cube['grid_shape']
    if not check_equal(cgrid_shape):
        raise ValueError("Cubic grid MUST be cubic, so same steps in each direction.")
    cvectors = cube['vectors']
    corigin = cube['origin']
    values = cube['values']
    cubicgrid = make_cubic_grid(cgrid_shape, cvectors, corigin)
    for i in range(3):
        cmin = corigin[i]
        cmax = corigin[i] + cgrid_shape[i]*cvectors[i, i]
        gmin = min(grid[:, i])
        gmax = max(grid[:, i])
        common_range.append((min(cmin, gmin), max(cmax, gmax)))
        div = cgrid_shape[i] // npoints
        spt = list(np.arange(0, cgrid_shape[i], div))
        vec = np.arange(cmin, cmax, cvectors[i][i])
        vec = vec[spt]
        # replace initial and final points if needed
        if vec[0] > common_range[i][0]:
            vec[0] = common_range[i][0]
        if vec[-1] < common_range[i][1]:
            vec[-1] = common_range[i][1]
        splits.append(vec)
    # Split grids in ranges
    # Use common minimum and maximum
    for b in range(len(splits[0]) - 1):  # number of blocks
        inds_cube = []
        inds_grid = []
        for i in range(3):
            rmin = splits[i][b]
            rmax = splits[i][b+1]
            # make masks for grids
       #    mins_cube = np.where(rmin <= cubicgrid[:, i])[0]
       #    maxs_cube = np.where(cubicgrid[:, i] <= rmax)[0]
            inds_cube.append(np.logical_and(rmin <= cubicgrid[:, i], cubicgrid[:, i] <= rmax))
            inds_grid.append(np.logical_and(rmin <= grid[:, i], grid[:, i] <= rmax))
       #    mins_grid = np.where(rmin <= grid[:, i])[0]
       #    maxs_grid = np.where(grid[:, i] <= rmax)[0]
        mask_cubic = [all(point) for point in list(zip(inds_cube[0], inds_cube[1], inds_cube[2]))]
        mask_grid = [all(point) for point in list(zip(inds_grid[0], inds_grid[1], inds_grid[2]))]
        grid[mask_grid] = interpolate_function(cubicgrid[mask_cubic],
                                               values[mask_cubic],
                                               grid[mask_grid])
    return grid


def check_equal(lst):
    return lst[1:] == lst[:-1]


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
