"""Script to make one single file trajectory, one fragment from xyz."""

import numpy as np
import shutil
import os
import subprocess as sp
import chemcoord as cc
from chemcoord.cartesian_coordinates._cartesian_class_get_zmat import CartesianGetZmat
import pandas as pd
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import time
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

t1=time.time()

def str2cart(xyz_str, start_index=1):
    """
    Note1
    ----
    StringIO must be imported differently according to the sys version.
    use: 
    import sys
    if sys.version_info[0] < 3: 
        from StringIO import StringIO
    else:
        from io import StringIO
        
    Note2
    -----
    StringIO can be used only once!
    
    Parameters
    ----------
    xyz_str: str
        xyz as a string.
    start_index: int
        starting point of the index for the DataFrame
    Returns
    -------
    chemcoordc.Cartesian
        chemcoord cartesian object (with bonds detected)
    """
    tofeed = StringIO(xyz_str)
    mol = cc.Cartesian.read_xyz(tofeed,start_index=start_index)
    return mol
    
def str2zmat(xyz_str, c_table, start_index=1):
    """
    Note1
    ----
    StringIO must be imported differently according to the sys version.
    use: 
    import sys
    if sys.version_info[0] < 3: 
        from StringIO import StringIO
    else:
        from io import StringIO
        
    Note2
    -----
    StringIO can be used only once!
    
    Parameters
    ----------
    xyz_str: str
        xyz as a string.
    start_index: int
        starting point of the index for the DataFrame
    Returns
    -------
    chemcoordc.Cartesian
        chemcoord cartesian object (with bonds detected)
    """
    tofeed = StringIO(xyz_str)
    mol = cc.Cartesian.read_xyz(tofeed,start_index=start_index)
    zmat = mol.get_zmat(c_table)
    return zmat

def conv_d(d):
    """
    """
    r = d % 360
    return r - (r // 180) * 360

def plot_dist(arr, index, title=""):
    x = np.arange(arr.shape[1])
    fig = plt.figure(figsize=(20,10), dpi = 150)
    ax = fig.add_subplot(111)
    ax.scatter(x, arr[index,:])
    ax.set_ylabel("dihedral")
    ax.set_xlabel("frame")
    ax.set_title(title)
    ax.set_ylim(-180, 180)
    return fig

def plot_dist_c(arr, index, title=""):
    x = np.arange(arr.shape[1])
    fig = plt.figure(figsize=(20,10), dpi = 150)
    ax = fig.add_subplot(111)
    ax.scatter(x,arr[index,:])
    ax.set_ylabel("dihedral")
    ax.set_xlabel("frame")
    ax.set_title(title)
    ax.set_ylim(0,360)
    return fig

fname = 'aligned0.xyz'
cwd = os.getcwd()
directory = os.path.join(cwd, 'retinal') 
final = os.path.join(cwd, 'zmat_averaged.xyz')

tot_lines = int(str(sp.check_output(["wc","-l",fname]))[2:].split()[0]) #getting total number of lines
with open(fname, 'r') as file:
    n_atoms = int(file.readline()[:-1]) #getting number of atoms reading only first line
assert tot_lines % (n_atoms+2) == 0 #checking tot_lines is an integer number of snapshots with n_atoms atoms
n_frames = tot_lines // (n_atoms+2)

bonds = np.zeros((n_atoms, n_frames))
angles = np.zeros((n_atoms, n_frames))
dih = np.zeros((n_atoms, n_frames))
with open(fname, 'r') as file: #really going to read
    for f in range(n_frames):
        frame_str = ""
        for i in range(n_atoms+2):
            frame_str += file.readline()
        if f == 0:
            cart = str2cart(frame_str)
            c_table = cart.get_construction_table()
        zmat = str2zmat(frame_str, c_table)
        bonds[:, f] = zmat._frame['bond']
        angles[:, f] = zmat._frame['angle']
        dih[:, f] = zmat._frame['dihedral']
bonds_m = bonds.mean(axis=1)
angles_m = angles.mean(axis=1)
dih_c = dih%360
dih_m = dih.mean(axis=1)  # averaged in the -180-180 range
dih_c_m = dih_c.mean(axis=1)  # averaged in the 0-360 range
dih_std = dih.std(axis=1)  # std when in the -180-180 range
dih_c_std = dih_c.std(axis=1) # std when in the 0-360range
dih_c_m_backup = dih_c_m.copy()
dih_c_m = conv_d(dih_c_m)  # reconvert to -180-180 range 
diff_std = dih_c_std-dih_std
use_c = np.where(diff_std<0) #where it's better to use 0-360 range
rotate = np.logical_and(abs(diff_std)<30,abs(dih_std)>90)
print("{} seem to be rotating".format(len(np.arange(60)[rotate])))
print(zmat._frame[rotate].index)
dihedrals_m = dih_m.copy()
dihedrals_m[use_c] = dih_c_m[use_c]

to_sub = np.array(zmat._frame["dihedral"])[rotate]
zmat._frame["bond"] = bonds_m
zmat._frame["angle"] = angles_m
zmat._frame["dihedral"] = dihedrals_m
carts = zmat.get_cartesian()
carts.to_xyz("averaged_quasi_straight.xyz")

dihedrals_m[rotate] = to_sub
zmat._frame["dihedral"]=dihedrals_m
carts = zmat.get_cartesian()
carts.to_xyz("averaged_rotate.xyz")
t2 = time.time()
elapsed=  t2-t1
print("elapsed time is {}".format(elapsed))
#carts.view(viewer="gmolden")
