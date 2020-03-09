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
from pyscf import gto, scf
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat
from pyscf.dft import gen_grid

from fdeta.fdetmd.mdinterface import MDInterface
from fdeta.mdtrajectory import MDTrajectory
from fdeta.fdetmd.interpolate import interpolate_function
from fdeta.fdetmd.interpolate import interpolate_function_split


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
    dic = os.getenv('FDETADATA')
    filetraj = os.path.join(dic, 'he_traj.txt')
    traj = MDTrajectory(filetraj)
    box_size = np.array([2, 2, 2])
    grid_size = np.array([5, 5, 5])
    # Use the mdinterface to create a cubic grid
    md = MDInterface(traj, box_size, grid_size)
  # print("Points: \n", md.points)
  # print(md.npoints*3)
    grid = np.zeros((md.npoints, 3))
    grid = md.pbox.get_grid(grid, False)

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
    # print(rho_all.shape)

    grids = gen_grid.Grids(mol0)
    grids.level = 4
    grids.build()
    # print(rho_all.shape)
    # print(grids.coords.shape)

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
    # print(minmax)
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
        # print("Point with problems", grids.coords[p[0]])
        # print(len(np.where(abs(new_rho1-real_rho1)>1e-1)))
      # fig = plt.figure()
      # ax = fig.add_subplot(projection='3d')
      # ax.scatter3D(xdata, ydata, new_rho1, c=new_rho1, cmap='Greens')
      # ax.scatter3D(xdata, ydata, real_rho1, c=real_rho1)
      # plt.xlim(-1.0,  1.0)
      # plt.ylim(-1.0, 1.0)
      # plt.show()
    # print(minmax1)


def test_interpolate_split():
    """Check interpolation for He density."""
    # Define variables
    dic = os.getenv('FDETADATA')
    traj = os.path.join(dic, 'traj_acetone_w2.xyz')
    box_size = np.array([30, 30, 30])
    grid_size = (15, 15, 15)
    ta = MDTrajectory(traj)
    mdi = MDInterface(ta, box_size, grid_size)
    ccoeffs = {'O': 1.1, 'H': 0.6}
    box_size2 = np.array([20, 20, 20])
    grid_size2 = (15, 15, 15)
    mdi2 = MDInterface(ta, box_size2, grid_size2)
    mdi.save_grid()
    gridname = "box_grid2.txt"
    mdi2.save_grid(gridname)
    box = np.loadtxt(gridname)
    gridn = np.zeros((len(box), 4))
    gridn[:, :3] = box[:]
    mdi.compute_electrostatic_potential(ccoeffs, gridn)
    vemb = np.loadtxt('elst_pot.txt')

    # Use PySCF to evaluate density
    with open("/home/cris/unige/projects/acetone/fdeta/2water/cc-pvdz-segopt.nwchem", 'r') as bf:
        basis = bf.read()
    mol0 = gto.M(atom="""C 0.000000000000 0.00000000000000  0.6391449000000001
                         C 0.000000000000 1.2830700000000006 -0.15244610000000008
                         C 0.000000000000 -1.2830700000000006 -0.15244610000000006
                         H 0.000000000000 2.1441410000000007 0.5110019000000001
                         H 0.000000000000 -2.1441410000000007 0.5110019000000001
                         H -0.8765800000000001 1.3168810000000004 -0.8034251000000002
                         H -0.8765799999999999 -1.3168810000000006 -0.8034251000000002
                         H 0.8765799999999999 1.3168810000000006 -0.8034251000000002
                         H 0.8765800000000001 -1.3168810000000004 -0.8034251000000001
                         O 0.000000000000000 0.0000000000000000 1.8574439000000003""",
                     basis=basis)

    # Solve HF and get density
    scfres = scf.RHF(mol0)
    scfres.conv_tol = 1e-12
    scfres.kernel()
    grids = gen_grid.Grids(mol0)
    grids.build()
    elst_pyscf = np.zeros((len(grids.weights), 4))
    elst_pyscf[:, :3] = grids.coords
#   elst_pyscf = interpolate_function_split(vemb[:, :3], vemb[:, 3], elst_pyscf,
#                                           function='inverse')
 #  np.savetxt("interpolated_elst_inverse.txt", elst_pyscf)
    elst_pyscf_direct = np.zeros(elst_pyscf.shape)
    elst_pyscf_direct[:, :3] = grids.coords
    mdi.compute_electrostatic_potential(ccoeffs, elst_pyscf_direct, fname='elst_pot_direct.txt')
    # Clean other files
    os.remove('elst_pot.txt')
    os.remove('aligned_0.xyz')
    os.remove('box_grid2.txt')


def integrate_interpolated():
    # Use PySCF to evaluate density
    with open("/home/cris/unige/projects/acetone/fdeta/2water/cc-pvdz-segopt.nwchem", 'r') as bf:
        basis = bf.read()
    mol0 = gto.M(atom="""C 0.000000000000 0.00000000000000  0.6391449000000001
                         C 0.000000000000 1.2830700000000006 -0.15244610000000008
                         C 0.000000000000 -1.2830700000000006 -0.15244610000000006
                         H 0.000000000000 2.1441410000000007 0.5110019000000001
                         H 0.000000000000 -2.1441410000000007 0.5110019000000001
                         H -0.8765800000000001 1.3168810000000004 -0.8034251000000002
                         H -0.8765799999999999 -1.3168810000000006 -0.8034251000000002
                         H 0.8765799999999999 1.3168810000000006 -0.8034251000000002
                         H 0.8765800000000001 -1.3168810000000004 -0.8034251000000001
                         O 0.000000000000000 0.0000000000000000 1.8574439000000003""",
                     basis=basis)

    # Solve HF and get density
    scfres = scf.RHF(mol0)
    scfres.conv_tol = 1e-12
    scfres.kernel()
    grids = gen_grid.Grids(mol0)
    grids.build()
    dm0 = scfres.make_rdm1()
    direct = np.loadtxt('elst_pot_direct.txt')
    ipd_gauss = np.loadtxt('interpolated_elst.txt')
    ipd_inv = np.loadtxt('interpolated_elst_inverse.txt')
    # integrate into AO matrices
    ao = eval_ao(mol0, grids.coords, deriv=0)
    rho = eval_rho(mol0, ao, dm0, xctype='LDA')
    vd = eval_mat(mol0, ao, grids.weights, rho, direct[:, 3], xctype='LDA')
    vg = eval_mat(mol0, ao, grids.weights, rho, ipd_gauss[:, 3], xctype='LDA')
    vi = eval_mat(mol0, ao, grids.weights, rho, ipd_inv[:, 3], xctype='LDA')
    np.savetxt('velst_direct.txt', vd*0.5, delimiter='\n')
    np.savetxt('velst_ipd_gauss.txt', vg*0.5, delimiter='\n')
    np.savetxt('velst_ipd_inv.txt', vi*0.5, delimiter='\n')



if __name__ == "__main__":
#   test_interpolation_base()
#   test_interpolate_helium()
#   test_interpolate_split()
    integrate_interpolated()
