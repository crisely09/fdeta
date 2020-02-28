"""Script to make one single file trajectory, one fragment from xyz."""

import sys
import numpy as np
import subprocess as sp
import chemcoord as cc
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib import rc
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


rc('text', usetex=True)


def str2cart(xyz_str: str, start_index: int = 1):
    """
    Note1
    ----
    StringIO must be imported differently according to the sys version.
    use :
    import sys
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else :
        from io import StringIO

    Note2
    -----
    StringIO can be used only once!

    Parameters
    ----------
    xyz_str : str
        xyz as a string.
    start_index : int
        starting point of the index for the DataFrame
    Returns
    -------
    chemcoordc.Cartesian
        chemcoord cartesian object (with bonds detected)
    """
    tofeed = StringIO(xyz_str)
    mol = cc.Cartesian.read_xyz(tofeed, start_index=start_index)
    return mol


def str2zmat(xyz_str, c_table, start_index=1):
    """
    Note1
    ----
    StringIO must be imported differently according to the sys version.
    use :
    import sys
    if sys.version_info[0] < 3 :
        from StringIO import StringIO
    else :
        from io import StringIO

    Note2
    -----
    StringIO can be used only once!

    Parameters
    ----------
    xyz_str : str
        xyz as a string.
    start_index : int
        starting point of the index for the DataFrame
    Returns
    -------
    chemcoordc.Cartesian
        chemcoord cartesian object (with bonds detected)
    """
    tofeed = StringIO(xyz_str)
    mol = cc.Cartesian.read_xyz(tofeed, start_index=start_index)
    zmat = mol.get_zmat(c_table)
    return zmat


def conv_d(d: int):
    """Convert degrees from (-180, 180) range
       to (0, 360) range.

    Parameters
    ----------
    d : int
        Degrees in (-180, 180) range.
    """
    r = d % 360
    return r - (r // 180) * 360


def plot_dist(data: np.ndarray, index: int,
              title: str = "", full_range=False):
    x = np.arange(data.shape[1])
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    ax.scatter(x, data[index, :])
    ax.set_ylabel("dihedral")
    ax.set_xlabel("frame")
    ax.set_title(title)
    if not full_range:
        ax.set_ylim(-180, 180)
    else:
        ax.set_ylim(0, 360)
    return fig


def correct_dihedrals(zmat: pd.frame, dih: np.darray):
    """ Find problematic dihedrals and correct them.

    Parameters
    ----------
    """
    dih_mean = dih.mean(axis=1)  # averaged in the -180-180 range
    dih_c = dih % 360
    dih_c_mean = dih_c.mean(axis=1)  # averaged in the 0-360 range
    dih_std = dih.std(axis=1)  # std when in the -180-180 range
    dih_c_std = dih_c.std(axis=1)  # std when in the 0-360range
    # dih_c_mean_backup = dih_c_mean.copy()
    dih_c_mean = conv_d(dih_c_mean)  # reconvert to -180-180 range
    diff_std = dih_c_std - dih_std
    use_c = np.where(diff_std < 0)  # where it's better to use 0-360 range
    # Find the freely rotating dihedrals
    rotate = np.logical_and(abs(diff_std) < 30, abs(dih_std) > 90)
    print("{} seem to be rotating".format(len(np.arange(60)[rotate])))
    print(zmat._frame[rotate].index)
    dihedrals_mean = dih_mean.copy()
    dihedrals_mean[use_c] = dih_c_mean[use_c]
    # Replace rotating angles with values of the last snapshot
    # TODO: Make sure it's not crazy
    to_sub = np.array(zmat._frame["dihedral"])[rotate]
    dihedrals_mean[rotate] = to_sub
    return dihedrals_mean


def get_averaged_structure_zmat(aligned_fn: str, averaged_fn: str):
    """Average a fragment using the internal coordinates.

    Parameters
    ----------
    aligned_fn : str
        Filename of the fragment aligned over all trajectory.
    averaged_fn : str
        Filename of the output averaged structure, in xyz format.
    """
    t1 = time.time()
    # Get total number of lines
    tot_lines = int(str(sp.check_output(["wc", "-l", aligned_fn]))[2:].split()[0])
    with open(aligned_fn, 'r') as file:
        # Get number of atoms reading only first line
        n_atoms = int(file.readline()[:-1])
    # Check if the total number of lines matches
    # the number of snapshots and atoms
    assert tot_lines % (n_atoms+2) == 0
    n_frames = tot_lines // (n_atoms+2)

    bonds = np.zeros((n_atoms, n_frames))
    angles = np.zeros((n_atoms, n_frames))
    dih = np.zeros((n_atoms, n_frames))
    with open(aligned_fn, 'r') as file:  # really going to read
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
    bonds_mean = bonds.mean(axis=1)
    angles_mean = angles.mean(axis=1)
    # Find problematic dihedrals and correct them
    dihedrals_mean = correct_dihedrals(zmat, dih)
    zmat._frame["bond"] = bonds_mean
    zmat._frame["angle"] = angles_mean
    zmat._frame["dihedral"] = dihedrals_mean
    carts = zmat.get_cartesian()
    carts.to_xyz(averaged_fn)
    t2 = time.time()
    elapsed = t2 - t1
    print("Elapsed time is {}".format(elapsed))
