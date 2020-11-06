
import os
import pytest
import numpy as np

from fdeta.cube import read_cubefile
from fdeta.mdft.io_tools import reduce_cube
from fdeta.mdft.io_tools import write_parameters_file, check_length
from fdeta.mdft.io_tools import read_parameters_file, check_length


def test_check_lengths():
    """Check if arrays have same length"""
    a = np.arange(10)
    b = np.arange(10, 20)
    c = np.arange(100).reshape((10, 10))
    d = np.arange(5)
    with pytest.raises(ValueError):
        check_length([a, d])
    with pytest.raises(ValueError):
        check_length([b, d])
    with pytest.raises(ValueError):
        check_length([c, d])
    check_length([a, b, c])


def test_read_write():
    """Read and write parameters file."""
    dic = os.getenv('FDETADATA')
    fname = os.path.join(dic, 'acetone_10sites')
    data = read_parameters_file(fname)
    c = """Acetone (propanone)   10-site model cooked by Daniel with """
    c += """geometry and charges from Cristina, and GAFF parameters for LJ\n"""
    assert c == data['comment']
    rcharge = np.array([0.85843, -0.52056, -0.52056, 0.14743, 0.14758, 0.131022, 0.131396,
                        0.130865, 0.130715, -0.635784])
    assert np.allclose(data['vcharge'], rcharge)
    write_parameters_file(data['elements'],
                          data['surnames'],
                          data['coords'],
                          data['vcharge'],
                          data['vsigma'],
                          data['vepsilon'])
    # Read new file and check for consistency
    data2 = read_parameters_file('pars.in')
    assert np.allclose(data['coords'], data2['coords'])
    assert (data['elements'] ==  data2['elements']).all()
    assert (data['surnames'] ==  data2['surnames']).all()
    assert np.allclose(data['vcharge'], data2['vcharge'])
    assert np.allclose(data['vsigma'], data2['vsigma'])
    assert np.allclose(data['vepsilon'], data2['vepsilon'])
    assert np.allclose(data['indices'], data2['indices'])
    os.remove('pars.in')


def test_reduce_cube():
    """Test for simple reduce_cube"""
    with pytest.raises(TypeError):
        reduce_cube(0, 0)
    # Read a cubefile
    dic = os.getenv('FDETADATA')
    name = 'solvent_charge_density_aroud_acetone.cube'
    fname = os.path.join(dic, name)
    tpoints = (13, 13, 13)
    with pytest.raises(ValueError):
        reduce_cube(fname, tpoints)
    tpoints = (20, 20, 20)
    newgrid, newvalues = reduce_cube(fname, tpoints)
    assert (newgrid.shape[0] == (80 - 20)**3)
    assert (newvalues.shape[0] == (80 - 20)**3)
    # Use cube data
    data = read_cubefile(fname)
    newgrid1, newvalues1 = reduce_cube(data, tpoints)
    assert (newgrid1.shape[0] == (80 - 20)**3)
    assert (newvalues1.shape[0] == (80 - 20)**3)
    assert np.allclose(newgrid, newgrid1)
    assert np.allclose(newvalues, newvalues1)


if __name__ == "__main__":
    test_check_lengths()
    test_read_write()
    test_reduce_cube()
