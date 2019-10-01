"""
Unit and regression test for the fdeta package.

"""

# Import package, test suite, and other packages as needed
import os
import numpy as np

from fdeta.fdetmd.dft import compute_nad_lda


def test_functionals():
    """ Test `MDInterface` functions for density functionals."""
    dic = os.getenv('FDETADATA')
    inp = np.loadtxt(os.path.join(dic, 'rhoA_ref.txt'))
    grida = inp[:, :3]
    rhoa = np.absolute(inp[:, 3])
    onp = np.loadtxt(os.path.join(dic, 'rhoB_ref.txt'))
    gridb = onp[:, :3]
    rhob = np.nan_to_num(onp[:, 3])
    assert np.allclose(grida, gridb)
    ec_nad, vc_nad = compute_nad_lda(rhoa*2.0, rhob*2.0, 'correlation')
    ex_nad, vx_nad = compute_nad_lda(rhoa*2.0, rhob*2.0, 'exchange')
    et_nad, vt_nad = compute_nad_lda(rhoa*2.0, rhob*2.0, 'kinetic')
    # Load references produced with Horton (libxc library)
    refs = np.load('vxct_nad.npz')
    assert np.allclose(vx_nad, refs['vx_nad'])
    assert np.allclose(vt_nad, refs['vt_nad'])
    assert np.allclose(vc_nad, refs['vc_nad'])


if __name__ == "__main__":
    test_functionals()
