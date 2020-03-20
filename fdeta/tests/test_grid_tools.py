"""
Unit and regression test for the fdeta package.

"""

# Import package, test suite, and other packages as needed
import numpy as np

from nose.tools import assert_raises
from fdeta.fdetmd.grid_tools import simplify_cubic_grid


def test_simplify_cubic_grid():
    """Basic test for function `simplify_cubic_grid`."""
    # Define variables to break code
    cubic = np.ones((4, 4))
    cubic_wrong_type = "no"
    cubic_wrong_size = np.ones((4,3))
    var = np.zeros((4, 4, 4))
    var_wrong_type = 59
    var_wrong_shape = np.ones((3, 4, 5))
    frames = 100
    frames_wrong = ['nah']
    assert_raises(TypeError, simplify_cubic_grid, cubic_wrong_type, var, frames)
    assert_raises(ValueError, simplify_cubic_grid, cubic_wrong_size, var, frames)
    assert_raises(TypeError, simplify_cubic_grid, cubic, var_wrong_type, frames)
    assert_raises(NotImplementedError, simplify_cubic_grid, cubic, var_wrong_shape, frames)
    assert_raises(TypeError, simplify_cubic_grid, cubic, var, frames_wrong)
