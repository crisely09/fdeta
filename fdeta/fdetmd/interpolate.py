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
    function : np.ndarray(N, dtype=float)
        Reference function values to create interpolator.
    grid : np.ndarray((n, 3), dtype=float)
        Grid where the interpolant will be evalated.
    function : string
        Name of method for the interpolation. Options are:
        `linear`, `cubic`, `gaussian`.

    """
    # Prepare arrays for interpolator
    xs = refgrid[:, 0]
    ys = refgrid[:, 1]
    zs = refgrid[:, 2]
    interpolator = interpolate.Rbf(xs, ys, zs, values, function=function)
    # Clear values previously stored
    grid[:, 3] = 0.0
    grid[:, 3] = interpolator(grid[:, :3])
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
