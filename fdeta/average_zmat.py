"""Script to make one single file trajectory, one fragment from xyz."""

import sys
import numpy as np
import subprocess as sp
import chemcoord as cc
import pandas as pd
import glob as gl
import time
import matplotlib.pyplot as plt
from matplotlib import rc
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


rc('text', usetex=True)


def plot_distrib(data: np.ndarray, index: int, bins=36,
                 y_label: str = "dihedral",  # TODO: allow plotting of multiple indexes together
                 title: str = "", pos_range: bool = True):
    """Plots value vs time
    """
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    range_ = (80, 360) if pos_range else (-180, 180)
    ax.hist(data[index, :], bins=bins, range=range_)
    ax.set_ylabel(y_label)  # oner may want to plot other parameters as well
    ax.set_xlabel("frame")
    ax.set_title(title)
    return fig


def plot_time_evo(data: np.ndarray, index: int, y_label: str = "dihedral",  # TODO: allow plotting of multiple indexes together
                  title: str = "", pos_range: bool = True):
    """Plots value vs time
    """
    x = np.arange(data.shape[1])
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    ax.scatter(x, data[index, :])
    ax.set_ylabel(y_label)  # oner may want to plot other parameters as well
    ax.set_xlabel("frame")
    ax.set_title(title)
    if not pos_range:
        ax.set_ylim(-180, 180)
    else:
        ax.set_ylim(0, 360)
    return fig


def str2cart(xyz_str: str, start_index: int = 1):
    """Gets cc.cartesian from string

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
    """Gets cc.zmat from string

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


def get_all_internal_coords(aligned_fn: str,
                            int_coords_file: str = "internal_coordinates.npz",
                            save: bool = True,
                            dec_digits: int = 3):
    """Retrieves all internal coordinates from aligned trajectory in cartesians.

    Parameters
    ----------
    aligned_fn : str
        Filename of the fragment aligned over all trajectory.
    int_coords_file : str
        Filename for writing all the internal coordinates. ".npy" and ".npz" can be
        detected, otherwise ".txt" separate files are used.
    save : bool
        whether coordinates should be also saved (True) or just returned (False)
    dec_digits : int
        Used only for ".txt" files, number of decimal digits

    Returns
    -------
    tuple(array, array, array)
        Bonds array(n_atoms,n_frames), Angles array(n_atoms,n_frames),
        Dihedrals array(n_atoms,n_frames)
    """

    t1 = time.time()
    # Get total number of lines
    tot_lines = int(str(sp.check_output(["wc", "-l", aligned_fn]))[2:].split()[0])
    with open(aligned_fn, 'r') as file:
        # Get number of atoms reading only first line
        n_atoms = int(file.readline()[:-1])
        if "n_atoms" in globals():  # i f variable is initiated but empty, will store it.
            globals()["n_atoms"] = n_atoms
    # Check if the total number of lines matches
    # the number of snapshots and atoms
    assert tot_lines % (n_atoms + 2) == 0
    n_frames = tot_lines // (n_atoms + 2)
    if "n_frames" in globals():  # i f variable is initiated but empty, will store it.
        globals()["n_frames"] = n_frames

    bonds = np.zeros((n_atoms, n_frames))
    angles = np.zeros((n_atoms, n_frames))
    dih = np.zeros((n_atoms, n_frames))
    with open(aligned_fn, 'r') as file:  # really going to read
        for f in range(n_frames):
            frame_str = ""
            for i in range(n_atoms + 2):
                frame_str += file.readline()
            if f == 0:
                cart = str2cart(frame_str)
                c_table = cart.get_construction_table()
                if "c_table" in globals():  # i f variable is initiated but empty, will store it.
                    globals()["c_table"] = c_table
                    zmat = str2zmat(frame_str, c_table)
                if "zmat" in globals():  # i f variable is initiated but empty, will store it.
                    globals()["zmat"] = zmat  # NB we stroe the first one!
            else:
                zmat = str2zmat(frame_str, c_table)
            bonds[:, f] = zmat._frame['bond']
            angles[:, f] = zmat._frame['angle']
            dih[:, f] = zmat._frame['dihedral']
        t2 = time.time()
    elapsed = t2 - t1
    print("Time to get all internal coordinates: {}".format(elapsed))
    if save:
        fmt = "{{%.f}}".format(dec_digits)
        if int_coords_file[:-4] == ".npy":
            np.save(int_coords_file, np.array([bonds, angles, dih]))
        elif int_coords_file[:-4] == ".npz":
            np.savez(int_coords_file, bonds=bonds, angles=angles, dihedrals=dih)
        else:
            int_coords_file += ".txt" if "txt" not in int_coords_file else ""
            np.savetxt(int_coords_file[:-4] + "_bonds.txt", np.array(bonds), fmt=fmt)
            np.savetxt(int_coords_file[:-4] + "_angles.txt", np.array(angles), fmt=fmt)
            np.savetxt(int_coords_file[:-4] + "_dihedrals.txt", np.array(dih), fmt=fmt)
            print("saved bonds,angles,dihedrals in {} respectively".format(", ".join(
                [int_coords_file[:-4] + i + int_coords_file[-4:] for i in["_bonds", "_angles", "_dihedrals"]])))
    return (bonds, angles, dih)


def correct_quasilinears(dih: np.darray = np.array([]),
                         dih_c: np.darray = np.array([]),
                         dih_std: np.darray = np.array([]),
                         dih_c_std: np.darray = np.array([])):
    """ Fixes dihedrals around + or - 180 by shifting to the 0-360 range, averaging, and back-converting.

    Parameters
    ----------
    dih: np.array(n_atoms,n_frames)
        array of dihedrals in the (-180,180) range.

    dih_c: np.array(n_atoms,n_frames)
        array of dihedrals in the (0,360) range. Default is to obtain it from dih
    dih_std: np.array(n_atoms)
        array of standard deviation for each dihedral in the (-180, 180) range. Default is to obtain it from dih
    dih_c_std: np.array(n_atoms)
        array of standard deviation for each dihedral in the (0,360) range. Default is to obtain it from dih_c
    Returns
    -------
    np.array(n_atoms)
        averaged dihedrals. NB rotation is not fixed yet.
    """

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

    if dih.size == 0:
        global dih
        dih = globals()["dih"]
        if dih.size == 0:
            raise IOError(r"there is no \dih\" array")
    global dih_mean, dih_c, dih_c_mean, dih_std, dih_c_std, use_c
    dih_mean = dih.mean(axis=1)  # averaged in the -180-180 range
    dih_c = dih % 360 if dih_c.size == 0 else dih_c
    dih_c_mean = dih_c.mean(axis=1)  # averaged in the 0-360 range
    dih_std = dih.std(axis=1) if dih_std.size == 0 else dih_std  # std when in the -180-180 range
    dih_c_std = dih_c.std(axis=1) if dih_c_std.size == 0 else dih_c_std  # std when in the 0-360range
    dih_c_mean = conv_d(dih_c_mean)  # reconvert to -180-180 range
    diff_std = dih_c_std - dih_std
    use_c = np.where(diff_std < 0)  # where it's better to use 0-360 range
    dih_mean[use_c] = dih_c_mean[use_c]
    return dih_mean


def detect_rotating_groups(dih_std: np.darray = np.array([]), dih_c_std: np.darray = np.array([])):
    """Find the freely rotating dihedrals
    Parameters
    ----------
    dih_std: np.array(n_atoms)
        array of standard deviation for each dihedral in the (0,360) range.
    dih_c_std: np.array(n_atoms)
        array of standard deviation for each dihedral in the (-180,180) range.

    Returns
    -------
    np.array(n_atoms,type=bool)
    """
    if dih_std.size == 0:
        try:
            global dih_std
            dih_std = globals()["dih"].std(axis=1)
        except BaseException:
            raise IOError(r"there is no \dih\" array")
    if dih_c_std.size == 0:
        try:
            global dih_c_std
            dih_c_std = globals()["dih_c"].std(axis=1)
        except BaseException:
            try:
                global dih_c
                global dih_c_std
                dih_c = globals()["dih"] % 360
                dih_c_std = dih_c.std(axis=1)
            raise IOError(r"there is no \dih\" array")
    rotate = np.logical_and(abs(dih_c_std - dih_std) < 30, abs(dih_std) > 90)
    print("{} atoms seem to be rotating".format(len(np.arange(60)[rotate])))
    return rotate


def average_int_coords(aligned_fn: str = "", int_coord_file: str = "",
                       out_file: str = "averaged.xyz", overwrite: bool = False,
                       view: bool = True, correct_quasilinears: bool = True,
                       detect_rotate: str = "std",
                       fix_rot: str = "pick_first"):
    """Averages the internal coordinates
    Parameters
    ----------
    aligned_fn: str
        aligned cartesian file name. Default is to search for default, if absent asks for input
    int_coord_file: str
        if provided it tries to interpret the extension, otherwise tries finding it or asks
    correct_quasilinears: bool
        whether quasilinear dihedrals should be corrected. Default is True
    detect_rotate: str
        method to detect rotation. std uses std in both (-180,180) and (0,360) range
        Other strings lead to no detection
    fix_rot: str
        method to select dihedrals for rotating groups. "Pick first" uses the dihedrals of the first frame

    Returns
    -------
    cc.Zmat
        the averaged structure
    """
    # 0 Some empty variables
    # they will all be updated if there is no int_coord_file
    global n_atoms, n_frames, c_table, zmat
    n_atoms, n_frames, c_table, zmat = 0, 0, None, None
    global bonds, angles, dih
    # 1 Reading or obtaining all internal coordinates
    # 1.1 Figuring out files present and extensions
    txtfiles = []
    if int_coord_file == "":
        ic_files = gl.glob("internal_coordinates.*")
        if len(ic_files) == 0:
            filesfound = [gl.glob("*_+" + i + ".txt") for i in ["bonds", "angles", "dihedrals"]]
            lenlist = list(map(len, filesfound))
            if lenlist == [1, 1, 1]:
                txtfiles = [i[0] for i in filesfound]
            elif 0 in lenlist:
                print("No internal_coordinate file found")
                try:
                    bonds, angles, dih = get_all_internal_coords(
                        aligned_fn="aligned0.xyz")  # use correct default name for aligned xyz
                except BaseException:
                    file_in = input("If the aligned .xyz exists, it did not have the default name. Please type it")
            else:
                raise FileNotFoundError("many bond,angles,dihedrals files. Specify \"file_in\" or clean up folder.")
        if len(ic_files) == 1:
            file_in = ic_files[0]
        if len(ic_files) > 1:
            file_in = input("There seem to be many internal coordinates files. Type the one to use.")

    if int_coord_file[-4:] == ".txt":
        try:
            txtfiles = [gl.glob(file_in[:-4] + "_+" + i + ".txt")[0] for i in ["bonds", "angles", "dihedrals"]]
        except BaseException:
            try:
                bonds, angles, dih = get_all_internal_coords(aligned_fn=file_in)
            except BaseException:
                raise FileNotFoundError(
                    "Cannot read it neither as cartesian trajectory nor as saved bonds,angles,dihedrals")
    if len(txtfiles) == 3:
        bonds = np.loadtxt(txtfiles[0])
        angles = np.loadtxt(txtfiles[1])
        dih = np.loadtxt(txtfiles[2])

    elif int_coord_file[-4:] == ".npz":
        bonds, angles, dih = [np.load(file_in)[i] for i in np.load(file_in)]
    elif int_coord_file[-4:] == ".npy":
        bonds, angles, dih = np.load(file_in)

    elif int_coord_file[-4:] == ".xyz":
        print("Nb, you gave the xyz as int_coord_file!")
        aligned_fn = int_coord_file
        int_coord_file = "internal_coordinates.npz"
        bonds, angles, dih = get_all_internal_coords(aligned_fn=aligned_fn)

    else:
        raise FileNotFoundError("Could not figure out the extension of your int_coord_file")
    # 1.2 If int_cord_file was read, reading beginning of aligned_fn for c_table, z_mat
    if n_atoms == 0:
        if aligned_fn == "":
            if os.path.exists("aligned0.xyz"):
                aligned_fn = "aligned0.xyz"  # use correct default name for aligned xyz
            else:
                file_in = input("If the aligned .xyz exists, it did not have the default name. Please type it")

        with open(aligned_fn, 'r') as file:
            n_atoms, n_frames = bonds.shape
            frame_str = ""
            for i in range(n_atoms + 2):
                frame_str += file.readline()
        cart = str2cart(frame_str)
        c_table = cart.get_construction_table()
        zmat = str2zmat(frame_str, c_table)
    first_zmat = zmat.copy()
    # 2 Averaging the internal coordinates
    global dih_mean, dih_c, dih_c_mean, dih_std, dih_c_std, use_c  # this way the first function setting them fixes them
    zmat._frame["bonds"] bonds.mean(axis=1)
    zmat._frame["angles"] angles.mean(axis=1)
    if correct_quasilinears == "std":
        correct_quasilinears()
    if detect_rotate == "std":
        rotate = detect_rotating_groups()
        if fix_rot == "pick_first":
            dih_mean[rotate] = np.array(first_zmat._frame["dihedrals"])[rotate]
    zmat._frame["dihedrals"] = dih_mean
    carts = zmat.get_cartesian()
    if overwrite or not os.path.exists(out_file):
        carts.to_xyz(out_file)
    else:
        splt = out_file.split(".")
        bname = ".".join(splt[:-1])
        ext = splt[-1]
        count = 1
        while os.path.exists(out_file):
            out_file = "{}_{}.{}".format(bname, count, ext)
            count + = 1
        carts.to_xyz(out_file)
    if view = True:
        carts.view()
