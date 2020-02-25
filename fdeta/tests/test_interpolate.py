"""
Unit and regression test for the fdeta package.

"""

# Import package, test suite, and other packages as needed
import os
import math
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fdeta.fdetmd.mdinterface import MDInterface
from fdeta.analysis import TrajectoryAnalysis


def test_interpolation_base():
    """Test interpolation."""

    def fun(x):
        """Test function"""
        return np.exp(x*np.cos(3*math.pi*x)) - 1.0

    x = np.arange(0, 1.1, 0.1)
    d = fun(x)
    rbfi = Rbf(x, d)
    y = np.arange(0., 1, 0.05)
    z = rbfi(y)
    t = fun(y)
    assert np.allclose(z, t, atol=1e-2)
  # plt.figure(figsize=(6.5, 4))
  # plt.plot(x, d, 'o', label='data')
  # plt.plot(y, t, label='true')
  # plt.plot(y, z, label='interpolated')
  # plt.legend()
  # plt.show()


def test_interpolate_helium():
    """Check interpolation for He density."""
    # define ta_object
    traj = TrajectoryAnalysis("he_traj.txt")
    box_size = np.array([2, 2, 2])
    grid_size = np.array([5, 5, 5])
    # Use the mdinterface to create a cubic grid
    md = MDInterface(traj, box_size, grid_size)
    print("Points: \n", md.points)
    print(md.npoints*3)
    grid = np.zeros((md.npoints, 3))
    grid = md.pbox.get_grid(grid)

    # Use PySCF to evaluate density
    from pyscf import gto, scf, dft
    from pyscf import lib
    from pyscf.dft.numint import eval_ao, eval_rho, eval_mat
    from pyscf.dft import gen_grid, libxc

    mol0 = gto.M(atom="""He  0.000   0.000   0.000""",
                     basis='sto-3g')

    # Solve HF and get density
    scfres = scf.RHF(mol0)
    scfres.conv_tol = 1e-12
    scfres.kernel()
    dm0 = scfres.make_rdm1()

    # Take only a plane in z=0
    subgrid = grid[50:75]
    ao_mol0 = eval_ao(mol0, subgrid, deriv=0)
    rho_mol0 = eval_rho(mol0, ao_mol0, dm0, xctype='LDA')
    rho_plot = rho_mol0.reshape((5, 5))

    # Check interpolation in PySCF grid
    from scipy import interpolate
    # whole grid
    ao_all = eval_ao(mol0, grid, deriv=0)
    rho_all = eval_rho(mol0, ao_all, dm0, xctype='LDA')
    xs = grid[:, 0]
    ys = grid[:, 1]
    zs = grid[:, 2]
    print(rho_all.shape)

    grids = gen_grid.Grids(mol0)
    grids.level = 4
    grids.build()
    print(rho_all.shape)
    print(grids.coords.shape)

    xdata = grids.coords[:, 0]
    ydata = grids.coords[:, 1]
    zdata = grids.coords[:, 2]

    # Real values
    real_ao = eval_ao(mol0, grids.coords, deriv=0)
    real_rho = eval_rho(mol0, real_ao, dm0, xctype='LDA')

    # Check with method is the best for Rbf interpolation
    functions = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']
    minmax = []
    for function in functions:
        print(function)
        interpolator = interpolate.Rbf(xs, ys, zs, rho_all, function=function)
        new_rho = interpolator(xdata, ydata, zdata)
        minmax.append([function, min(abs(new_rho - real_rho)), max(abs(new_rho - real_rho))])
      # fig = plt.figure()
      # ax = fig.add_subplot(projection='3d')
      # ax.scatter3D(xdata, ydata, new_rho, c=new_rho, cmap='Greens')
      # ax.scatter3D(xdata, ydata, real_rho, c=real_rho)
      # plt.xlim(-1.0,  1.0)
      # plt.ylim(-1.0, 1.0)
      # plt.show()
    print(minmax)
    mol1 = gto.M(atom="""He  0.000   0.000   2.500""",
                     basis='sto-3g')
    # Solve HF and get density
    scfres1 = scf.RHF(mol1)
    scfres1.conv_tol = 1e-12
    scfres1.kernel()
    dm1 = scfres1.make_rdm1()

    # whole grid
    ao_all1 = eval_ao(mol1, grid, deriv=0)
    rho_all1 = eval_rho(mol1, ao_all1, dm1, xctype='LDA')
    # Real values
    real_ao1 = eval_ao(mol1, grids.coords, deriv=0)
    real_rho1 = eval_rho(mol1, real_ao1, dm1, xctype='LDA')
    minmax1 = []
    for function in functions:
        interpolator = interpolate.Rbf(xs, ys, zs, rho_all1, function=function)
        new_rho1 = interpolator(xdata, ydata, zdata)
        minmax1.append([function, min(abs(new_rho1 - real_rho1)), max(abs(new_rho1 - real_rho1))])
        p = np.where(abs(new_rho1-real_rho1) == minmax1[-1][2])
        print("Point with problems", grids.coords[p[0]])
        print(len(np.where(abs(new_rho1-real_rho1)>1e-1)))
      # fig = plt.figure()
      # ax = fig.add_subplot(projection='3d')
      # ax.scatter3D(xdata, ydata, new_rho1, c=new_rho1, cmap='Greens')
      # ax.scatter3D(xdata, ydata, real_rho1, c=real_rho1)
      # plt.xlim(-1.0,  1.0)
      # plt.ylim(-1.0, 1.0)
      # plt.show()
    print(minmax1)



if __name__ == "__main__":
#   test_interpolation_base()
    test_interpolate_helium()
