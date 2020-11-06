"""

Density Functionals on a Grid.

"""

import numpy as np


def compute_kinetic_tf(rho):
    """Thomas-Fermi kinetic energy functional."""
    cf = (3./10.)*(np.power(3.0*np.pi**2, 2./3.))
    et = cf*(np.power(rho, 5./3.))
    vt = cf*5./3.*(np.power(rho, 2./3.))
    return et, vt


def compute_exchage_slater(rho):
    """Slater exchange energy functional."""
    from scipy.special import cbrt
    cx = (3./4) * (3/np.pi)**(1./3)
    ex = - cx * (np.power(rho, 4./3))
    vx = -(4./3) * cx * cbrt(np.fabs(rho))
    return ex, vx


def compute_corr_pyscf(rho, xc_code=',VWN'):
    """Correlation energy functional."""
    from pyscf.dft import libxc
    exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho)
    return exc, vxc[0]


def compute_nad_lda(rhoa, rhob, func):
    """Compute the non-additive kinetic energy functional and potential.

    Parameters
    ----------
    rhoa, rhob : np.ndarray
        Densities of fragments A and B on a grid.
    func : str
        Name of density functional to use.
        Options are:`kinetic` for Thomas-Fermi, `exchange` for Slater and
        `correlation` for VWN5.
    """
    rho_tot = rhoa + rhob
    if func == 'kinetic':
        functional = compute_kinetic_tf
    elif func == 'exchange':
        functional = compute_exchage_slater
    elif func == 'correlation':
        functional = compute_corr_pyscf
    else:
        raise ValueError('Only `kinetic`, `exchange` or `correlation` options are valid.')
    e, v = functional(rho_tot)
    ea, va = functional(rhoa)
    eb, vb = functional(rhob)
    etot = e - ea - eb
    vtot = v - va
    return etot, vtot


def compute_nad_lda_all(rhoa, rhob):
    """Compute the whole LDA embedding potential."""
    enad = np.zeros(rhoa.shape)
    vnad = np.zeros(rhoa.shape)
    for func in ['kinetic', 'exchange', 'correlation']:
        etmp, vtmp = compute_nad_lda(rhoa, rhob, func)
        enad += etmp
        vnad += vtmp
    return enad, vnad
