"""Script to make one single file trajectory, one fragment from xyz."""

from typing import Union
import os
import numpy as np
import subprocess as sp
import chemcoord as cc
import pandas as pd
import glob as gl
import time
import itertools as it
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


rc('text', usetex=True)
def plot_2Ddistrib(data: np.ndarray, index: list, bins: int = 36,
                 labels: list= [],  
              title: str = "", pos_range: bool = True):
    """Plots value occurrence vs value
    Parameters
    ----------
    data: np.array
        the array to slice and plot from. Generally np.array(natoms, nframes)
    index: list or int
        index or list of indexes to plot (data[index,:])
    bins: int
        number of bins for histogram. default is 36
    y_label: str 
        label for y axis. default is "dihedral"
    title: str
        title for the plot. default is empty string
    pos_range: bool
        whether to plot in the (-180,180) or in the (0,360) range
    
    Returns
    -------
    plt.figure
        the histogram with distribution of data[i,:] for i in index
    """
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    range_ = (0, 360) if pos_range else (-180, 180)
    ax.hist2d(data[index[0], :], data[index[1], :], bins=[bins,bins], range=(range_,range_))
    if not labels:
        ax.set_xlabel("{}".format(index[0]))
        ax.set_ylabel("{}".format(index[1]))
    else:
        ax.set_xlabel("{}".format(labels[0]))
        ax.set_ylabel("{}".format(labels[1]))
    ax.set_title(title)
    return fig

def plot_distrib(data: np.ndarray, index: Union[list,int], bins: int = 36,
                 x_label: str = "dihedral",  
              title: str = "", pos_range: bool = True):
    """Plots value occurrence vs value
    Parameters
    ----------
    data: np.array
        the array to slice and plot from. Generally np.array(natoms, nframes)
    index: list or int
        index or list of indexes to plot (data[index,:])
    bins: int
        number of bins for histogram. default is 36
    y_label: str 
        label for y axis. default is "dihedral"
    title: str
        title for the plot. default is empty string
    pos_range: bool
        whether to plot in the (-180,180) or in the (0,360) range
    
    Returns
    -------
    plt.figure
        the histogram with distribution of data[i,:] for i in index
    """
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    range_ = (0, 360) if pos_range else (-180, 180)
    if type(index) == int:
        index = [index]
    for i in index:
        ax.hist(data[i, :], bins=bins, range=range_)
    ax.set_ylabel("occurrence") 
    ax.set_xlabel(x_label)
    ax.set_title(title)
    return fig


def plot_time_evo(data: np.ndarray, index: Union[list,int], y_label: str = "dihedral",  
                  title: str = "", pos_range: bool = True):
    """Plots value vs time
    Parameters
    ----------
    data: np.array
        the array to slice and plot from. Generally np.array(natoms, nframes)
    index: list or int
        index or list of indexes to plot (data[index,:])
    y_label: str 
        label for y axis. default is "dihedral"
    title: str
        title for the plot. default is empty string
    pos_range: bool
        whether to plot in the (-180,180) or in the (0,360) range
    
    Returns
    -------
    plt.figure
        the value vs t of data[i,:] for i in index
    """
    x = np.arange(data.shape[1])
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    if type(index) == int:
        index = [index]
    for i in index:
        ax.scatter(x, data[i, :])
    ax.set_ylabel(y_label)  
    ax.set_xlabel("frame")
    ax.set_title(title)
    if not pos_range:
        ax.set_ylim(-180, 180)
    else:
        ax.set_ylim(0, 360)
    return fig

class ic_averager:
    """Object for averaging of molecule A in internal coordinates.
    Allows different methods to detect and fix issues such as quasilinear angles and rotating groups.
    """

    def __init__(self,  bonds: np.ndarray = np.array([]), angles: np.ndarray = np.array([]),
                 dih: np.ndarray = np.array([]), zmat1: Union[cc.Zmat, None] = None,  #todo check Union
                 c_table: pd.DataFrame = pd.DataFrame()):
        """
        Parameters
        ----------
        bonds: np.array
            either (natoms,nframes) or already averaged (natoms)
        angles: np.array
            either (natoms,nframes) or already averaged (natoms)
        dih: np.array(natoms,nframes)
            always all dihedrals for each frame
        zmat1: cc.Zmat
            initial zmat, for reference
        c_table: pd.DataFrame
            conversion table from cart1 to zmat1
        """
        self.bonds = bonds
        self.angles = angles
        self.dih = dih
        self.zmat1 = zmat1
        self.c_table = c_table
        self.zmat = None
        self.cart = None
        self.natoms = self.dih.shape[0] 
        self.nframes = self.dih.shape[1]
        if len(self.bonds.shape)==0:
            self.avg_bond_angles = None  # empty attributes, so no info on averaging
        elif len(self.bonds.shape)==1:
            self.avg_bond_angles = True  # bonds and angles are already averaged
        elif len(self.bonds.shape)==2:
            self.avg_bond_angles = False # full distribution of bonds and angles
        else:
            raise AttributeError("Wrong shape for bond array")
            
    def copy(self):
        """Copies the ic_averager        
        """
        import copy as c
        return c.copy(self)
        
    @classmethod
    def from_arrays(cls, atoms: Union[np.ndarray,list], arr: np.ndarray, int_coords_file: str = "internal_coordinates.npz",
                               save: bool =True, avg_bond_angles: bool = False, dec_digits: int = 3):
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
        avg_bond_angles: bool
            whether bonds and angles should already be averaged (for saving and returning)
        dec_digits : int
            Used only for ".txt" files, number of decimal digits
        """
    
        t1 = time.time()
        nframes, natoms =  arr.shape[:2]
        bonds = np.zeros((natoms, nframes))
        angles = np.zeros((natoms, nframes))
        dih = np.zeros((natoms, nframes))
        for f in range(nframes):
            df = pd.DataFrame(arr[f], columns=["x","y","z"])
            df.insert(0, "atom", atoms)
            cart = cc.Cartesian(df)
            if f == 0:
                c_table = cart.get_construction_table()
                zmat = cart.get_zmat(c_table)
                zmat1 = zmat.copy()
            else:
                zmat = cart.get_zmat(c_table)
                bonds[:, f] = zmat._frame['bond']
                angles[:, f] = zmat._frame['angle']
                dih[:, f] = zmat._frame['dihedral']
            t2 = time.time()
        elapsed = t2 - t1
        print("Time to get all internal coordinates: {}".format(elapsed))
        if save:
            if avg_bond_angles:
                bonds = bonds.mean(axis = 1)
                angles = angles.mean(axis = 1)
            fmt = "{{%.f}}".format(dec_digits)
            if int_coords_file[-4:] == ".npy":
                np.save(int_coords_file, np.array([bonds, angles, dih]))
                print("saved bonds, angles, dihedrals in {}".format(int_coords_file))
            elif int_coords_file[-4:] == ".npz":
                np.savez(int_coords_file, bonds=bonds, angles=angles, dihedrals=dih)
                print("saved bonds, angles, dihedrals in {}".format(int_coords_file))
            else:
                int_coords_file += ".txt" if "txt" not in int_coords_file else ""
                np.savetxt(int_coords_file[:-4] + "_bonds.txt", np.array(bonds), fmt=fmt)
                np.savetxt(int_coords_file[:-4] + "_angles.txt", np.array(angles), fmt=fmt)
                np.savetxt(int_coords_file[:-4] + "_dihedrals.txt", np.array(dih), fmt=fmt)
                print("saved bonds,angles,dihedrals in {} respectively".format(", ".join(
                    [int_coords_file[:-4] + i + int_coords_file[-4:] for i in["_bonds", "_angles", "_dihedrals"]])))
        return cls(bonds, angles, dih, zmat1, c_table)
#       
    @classmethod
    def from_aligned_cartesian_file(cls, aligned_fn: str = "aligned0.xyz", int_coords_file: str = "internal_coordinates.npz",
                               save: bool =True, avg_bond_angles: bool = False, dec_digits: int = 3):
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
        avg_bond_angles: bool
            whether bonds and angles should already be averaged (for saving and returning)
        dec_digits : int
            Used only for ".txt" files, number of decimal digits
        """
    
        t1 = time.time()
        # Get total number of lines
        tot_lines = int(str(sp.check_output(["wc", "-l", aligned_fn]))[2:].split()[0])
        with open(aligned_fn, 'r') as file:
            # Get number of atoms reading only first line
            natoms = int(file.readline()[:-1])
        # Check if the total number of lines matches
        # the number of snapshots and atoms
        assert tot_lines % (natoms + 2) == 0
        nframes = tot_lines // (natoms + 2)
    
        bonds = np.zeros((natoms, nframes))
        angles = np.zeros((natoms, nframes))
        dih = np.zeros((natoms, nframes))
        with open(aligned_fn, 'r') as file:  # really going to read
            for f in range(nframes):
                frame_str = ""
                for i in range(natoms + 2):
                    frame_str += file.readline()
                if f == 0:
                    cart = str2cart(frame_str)
                    c_table = cart.get_construction_table()
                    zmat = str2zmat(frame_str, c_table)
                    zmat1 = zmat.copy()
                else:
                    zmat = str2zmat(frame_str, c_table)
                bonds[:, f] = zmat._frame['bond']
                angles[:, f] = zmat._frame['angle']
                dih[:, f] = zmat._frame['dihedral']
            t2 = time.time()
        elapsed = t2 - t1
        print("Time to get all internal coordinates: {}".format(elapsed))
        if save:
            if avg_bond_angles:
                bonds = bonds.mean(axis = 1)
                angles = angles.mean(axis = 1)
#            fmt = "{{%.f}}".format(dec_digits)
            if int_coords_file[-4:] == ".npy":
                np.save(int_coords_file, np.array([bonds, angles, dih]))
                print("saved bonds, angles, dihedrals in {}".format(int_coords_file))
            elif int_coords_file[-4:] == ".npz":
                np.savez(int_coords_file, bonds=bonds, angles=angles, dihedrals=dih)
                print("saved bonds, angles, dihedrals in {}".format(int_coords_file))
            else:
                int_coords_file += ".txt" if "txt" not in int_coords_file else ""
                np.savetxt(int_coords_file[:-4] + "_bonds.txt", np.array(bonds))
                np.savetxt(int_coords_file[:-4] + "_angles.txt", np.array(angles))
                np.savetxt(int_coords_file[:-4] + "_dihedrals.txt", np.array(dih))
                print("saved bonds,angles,dihedrals in {} respectively".format(", ".join(
                    [int_coords_file[:-4] + i + int_coords_file[-4:] for i in["_bonds", "_angles", "_dihedrals"]])))
        return cls(bonds, angles, dih, zmat1, c_table)
    
    @classmethod
    def from_int_coord_file(cls, aligned_fn: str = "", int_coord_file: str = ""):
        """ Retrieves all internal coordinates from file
        Parameters
        ----------
        aligned_fn : str
            Filename of the fragment aligned over all trajectory. Needed for zmat1, c_table
        int_coords_file : str
            File with all the internal coordinates. ".npy" and ".npz" can be
            detected, otherwise ".txt" separate files are used.
        """
        # 1.1 Figuring out files present and extensions
        txtfiles = []
        processed = False
        if int_coord_file == "":
            print("No internal coordinates file specified. trying to figure it out.")
            ic_files = gl.glob("internal_coordinates.*")
            if len(ic_files) == 0:
                filesfound = [gl.glob("*_" + i + ".txt") for i in ["bonds", "angles", "dihedrals"]]
                lenlist = list(map(len, filesfound))
                if lenlist == [1, 1, 1]:
                    txtfiles = [i[0] for i in filesfound]
                    print("Using {}".format(", ".join(txtfiles)))
                elif 0 in lenlist:
                    raise FileNotFoundError("Could not find the internal coordinates.")
                else:
                    raise FileNotFoundError("many bond,angles,dihedrals files. Specify \"int_coord_file\" or clean up folder.")
            if len(ic_files) == 1:
                int_coord_file = ic_files[0]
                print("Using {}".format(int_coord_file))
            if len(ic_files) > 1:
                int_coord_file = input("There seem to be many internal coordinates files. Type the one to use:\n")
    
        if int_coord_file[-4:] == ".txt":
            try:
                txtfiles = [gl.glob(int_coord_file[:-4] + "_" + i + ".txt")[0] for i in ["bonds", "angles", "dihedrals"]]
            except BaseException:
                raise FileNotFoundError(
                        """For *.txt, internal coordinates should be saved as basename_bonds.txt , basename_angles.txt, basename_dihedrals.txt.\n 
                        Feed to function int_coord_file=\"basename.txt\
                        """ )
        if len(txtfiles) == 3:
            bonds = np.loadtxt(txtfiles[0])
            angles = np.loadtxt(txtfiles[1])
            dih = np.loadtxt(txtfiles[2])
            processed = True
        elif len(txtfiles) == 0:
            pass
        else:
            raise FileNotFoundError(""""For *.txt, internal coordinates should be saved as basename_bonds.txt , basename_angles.txt, basename_dihedrals.txt.\n 
                        Feed to function int_coord_file=\"basename.txt\
                        """ )
    
        if int_coord_file[-4:] == ".npz":
            bonds, angles, dih = [np.load(int_coord_file)[i] for i in np.load(int_coord_file)]
        elif int_coord_file[-4:] == ".npy":
            bonds, angles, dih = np.load(int_coord_file)
        elif not processed:
            raise FileNotFoundError("Could not figure out the extension of your int_coord_file")
        # 1.2 If int_cord_file was read, reading beginning of aligned_fn for c_table, z_mat
        if aligned_fn == "":
            if os.path.exists("aligned0.xyz"):
                aligned_fn = "aligned0.xyz"  # use correct default name for aligned xyz
            else:
                aligned_fn = input("If the aligned .xyz exists, it did not have the default name. Please type it\n")

        with open(aligned_fn, 'r') as file:
            n_atoms, n_frames = bonds.shape
            frame_str = ""
            for i in range(n_atoms + 2):
                frame_str += file.readline()
        cart = str2cart(frame_str)
        c_table = cart.get_construction_table()
        zmat1 = str2zmat(frame_str, c_table)
        return cls(bonds, angles, dih, zmat1, c_table)
    
    def correct_quasilinears(self, method: str = "std"):
        """ Fixes wrong averaging for dihedrals around + or - 180.
    
        Parameters
        ----------
        method: str
            method to correct quasilinears. Options:
                "std": shifts to the 0-360 range, and if std is smaller, averages and back-converts
        
        Sets
        ----
        self.use_c: np.array
            dihedrals where the (0,360) range should be used
        self.zmat:
            updates dihedrals in zmat(avg)
        """
        if method == "std":
            if not hasattr(self,"use_c"):
                diff_std = self.dih_c_std - self.dih_std
                self.use_c = np.where(diff_std < 0)[0]  # where it's better to use 0-360 range
            try:
                self.zmat._frame["dihedral"].values[self.use_c] = self.dih_c_mean[self.use_c]
            except:
                print("zmat not defined yet, so only setting self.use_c")
    
    def detect_rotating_groups(self, method: str = "std", thresh_min: Union[float, int] = 90, thresh_max: Union[float, int] = 180, group_name: str = "rotate"):
        """Find the freely rotating dihedrals
        Parameters
        ----------
        method: str
            method to detect dihedrals. Options:
                "std": dih for which the minimum std deviation between (-180,180) and (0,360) is between thres_min and thresh_max
        thresh_min: float or int
            threshold for minimum std for the dihedral on the most appropriate range
        thresh_max: float or int
            threshold for maximum std for the dihedral on the most appropriate range
                
        Sets
        ----
        self.[group_name]: np.array
            indexes of rotating dihedral. Nb this is numpy simple (0-natoms) numbering for dih,
            not pandas zmat._frame.index, i.e. the cartesian numbering.
        """
        if method == "std":
            detected = np.arange(3,self.natoms)[np.logical_and(np.minimum(self.dih_std[3:], self.dih_c_std[3:]) > thresh_min,np.minimum(self.dih_std[3:], self.dih_c_std[3:]) < thresh_max)]
            
            setattr(self, group_name, detected)
            print("{} atoms have been detected with the specified thresholds".format(detected.shape[0]))  
            print("array numbering: {}".format(", ".join(list(map(str,detected)))))
            print("Cartesian numbering: {}".format(", ".join(list(map(str,self.zmat1._frame.index[detected])))))
    
    def find_clusters(self, var: str = "", index: Union[None, int] = None, min_centers: int = 1, max_centers: int =3):
        """
        TODO:docstring
        """
        using_c = False
        if index == None:
            raise ValueError("You must specify \"index\".")
        if var == "":
            print("no variable type specified, supposing it is \"dihedral\"")
            var = "dihedral"
        if var == "dihedral":
            if hasattr(self, "use_c"):
                (arr, using_c) = (self.dih_c[index],True) if index in self.use_c else (self.dih[index], False)
            else:
                print("Watch out! No correction of quasilinear dihedrals has been performed thus far!")
                arr = self.dih
            clustdictname = "dihclustdict"
            centdictname = "dihcentdict"
        elif var == "angle":
            arr = self.angles[index]
            clustdictname = "angleclustdict"
            centdictname = "anglecentdict"
        elif var == "bond":
            arr = self.bonds[index]
            clustdictname = "bondclustdict"
            centdictname = "bondcentdict"
        else: 
            raise ValueError("Use either \"dihedral\" or \"angle\" or \"bond\".")
        arr = arr.reshape(-1,1)
        initial_centers = kmeans_plusplus_initializer(arr, min_centers).initialize()
        xmeans_instance = xmeans(arr, initial_centers, max_centers)
        xmeans_instance.process()
        if hasattr(self, clustdictname):
            dict_ = getattr(self, clustdictname)
        else:
            dict_ = {}
        dict_[index] = list(map(np.asarray, xmeans_instance.get_clusters()))
        setattr(self, clustdictname, dict_)
        if hasattr(self, centdictname):
            dict_ = getattr(self, centdictname)
        else:
            dict_ = {}
        centers = [conv_d(i) for i in list(it.chain.from_iterable(xmeans_instance.get_centers()))] if using_c else list(it.chain.from_iterable(xmeans_instance.get_centers()))
        dict_[index] = list(map(np.asarray, centers))
        setattr(self, centdictname, dict_)
        

    def fix_rotate(self, method: str = "pick_first", group_name: str = "rotate"):
        """Fixes the issue of rotating groups.
        Parameters
        ----------
        method: str
            method to use. Options:
                "pick_first": pick dihedrals for self.rotate from the first frame
                "major_basin": detects groups of atoms bound to the same atom (e.g. methyl), 
                               analyses the one among these that has most clear major/minor conformations
                               selects the major one, and applies averaged difference to the other angles
                "minor_basin": the same as major but selects the minor basin                                
        """
        if method == "pick_first":
            self.zmat._frame["dihedral"].values[getattr(self, group_name)] = self.zmat1._frame["dihedral"].values[getattr(self, group_name)]
        if method in ["major_basin","minor_basin"]:
            groups=[]  # becomes list of 1D-arrays
            for r  in getattr(self, group_name):
                if r not in list(it.chain.from_iterable(groups)):
                    b2s_cart = self.c_table[self.c_table["b"]==self.c_table.iloc[r]["b"]].index  # bound to the same, cartesian numbering
                    b2s_arr = [self.c_table.index.get_loc(i) for i in b2s_cart]  # passing to arr numbering
                    groups.append(b2s_arr)
            for gr in groups:  # gr is an array
                if len(gr)==1: # if only 1 rotating in gr,  e.g. H in OH
                    if not hasattr(self,"dihclustdict") or gr[0] not in self.dihclustdict.keys():
                        self.find_clusters(index = gr[0], min_centers = 2, max_centers = 3, var = "dihedral")
                    weights = [len(i) for i in self.dihclustdict[gr[0]]]
                    main = weights.index(max(weights)) if method=="major_basin" else weights.index(sorted(weights)[-2])
                    self.zmat._frame["dihedral"].values[gr[0]] = self.dihcentdict[gr[0]][main]
                else:  # if more than 1 rotating in gr, e.g. methyl
                    weights = []  # will be list of lists with basin weights per dih in group
                    sort_weights = []  # same but ordered
                    prom = []  # prominence for each d in gr
                    for n,d in enumerate(gr):  # n is numbering within group, d is numbering for dih
                        if not hasattr(self,"dihclustdict") or d not in self.dihclustdict.keys():
                            self.find_clusters(index = d, min_centers = 2, max_centers = 3, var = "dihedral")
                        weights.append([len(i) for i in self.dihclustdict[d]])
                        sort_weights.append(sorted(weights[n]))
                        prom.append(sort_weights[n][-1] - sort_weights[n][-2])
                    topick = prom.index(max(prom))  # best n to pick the major basin, use gr[topick] to get d
                    main = weights[topick].index(max(weights[topick])) if method=="major_basin" else weights[topick].index(sorted(weights[topick])[-2])
                    tosub = self.dihcentdict[gr[topick]][main]
                    for n,d in enumerate(gr):  # n is numbering within group, d is numbering for dih
                        if n == topick:
                            self.zmat._frame["dihedral"].values[d] = tosub 
                        else:
                            diff = (conv_d(self.dih[gr[topick],:] - self.dih[d,:])).mean()
                            self.zmat._frame["dihedral"].values[d] = conv_d(tosub - diff)
                        
    def average_int_coords(self, out_file: str = "averaged.xyz", overwrite: bool = False,
                           view: bool = True, viewer: str = "avogadro",
                           correct_quasilinears: str = "std",
                           detect_rotate: str = "std",
                           thresh_rot: Union[float, int] = 90,
                           fix_rotate: str = "pick_first",
                           detect_osc: str = "std",
                           thresh_osc: Union[float, int] = 37.5,
                           fix_osc: str = "major_basin"):
        """Averages the internal coordinates. Rotate/ing refers to fully/freely rotating groups,
        oscillate/ing refers to groups oscillating between two conformers.
        
        Parameters
        ----------
        out_file: str
            where to write the averaged structure. default is "averaged.xyz"
        overwrite: bool
            whether out_file can be overwritten. if False it creates averaged_n.xyz with n=1,2...
        correct_quasilinears: str
            how quasilinear dihedrals should be corrected. "no" to skip
        detect_rotate: str
            method to detect rotation. "no" to skip
        thresh_rot: float or int
            used as thresh_min for self.detect_rotate
        fix_rot: str
            method to select dihedrals for rotating groups. "no" to skip"
        detect_osc: str
            method to detect rotation. "no" to skip
        thresh_osc: float or int
            used as thresh_min for self.detect_rotate(group_name="oscillate"), thresh_max is set to thresh_rot
        fix_osc: str
            method to select dihedrals for oscillating groups. "no" to skip"    
        Returns
        -------
        cc.Zmat
            the averaged structure
        """
        print("tresholds: {}, {}".format(thresh_osc,thresh_rot))
        options = {"correct_quasilinears": ["no", "std"],
                   "detect_rotate": ["no", "std", "stored"],
                   "fix": ["no", "pick_first", "major_basin","minor_basin"]}
        if correct_quasilinears not in options["correct_quasilinears"]:
            raise ValueError("correct_quasilinears can be among {}".format(", ".join(options["correct_quasilinears"])))
        if detect_rotate not in options["detect_rotate"]:
            raise ValueError("detect_rotate can be among {}".format(", ".join(options["detect_rotate"])))
        if fix_rotate not in options["fix"] or fix_osc not in options["fix"]:
            raise ValueError("fixing methods can be among {}".format(", ".join(options["fix"])))
            
        self.zmat = self.zmat1.copy()
        self.zmat._frame["bonds"] == self.bonds if self.avg_bond_angles else self.bonds.mean(axis=1) 
        self.zmat._frame["angles"] == self.angles if self.avg_bond_angles else self.angles.mean(axis=1)
        self.zmat._frame["dihedral"] = self.dih_mean
        
        if correct_quasilinears != "no":
            self.correct_quasilinears(correct_quasilinears)
#            print("corrected quasilinears")
        if detect_rotate != "no":  # all acting on group_name="rotate", which is default
            if detect_rotate == "std":
                self.detect_rotating_groups(method = "std", thresh_min = thresh_rot)
            elif detect_rotate == "stored":
                pass
            else:
                raise ValueError("Only \"std\" is implemented now as rotation detection")
            if fix_rotate != "no":
                self.fix_rotate(method = fix_rotate)
#                print("fixed rotate with method: {}".format(fix_rotate))
        if detect_osc != "no":  # all acting on group_name="osc"
            if detect_osc == "std":
                self.detect_rotating_groups(method = "std", thresh_min = thresh_osc, thresh_max = thresh_rot, group_name = "osc")
            elif detect_osc == "stored":
                pass
            else:
                raise ValueError("Only \"std\" is implemented now as rotation detection")
            if fix_osc != "no":
                print("about to fix osc")
                self.fix_rotate(method = fix_osc, group_name = "osc")  
#                print("fixed oscillate with method: {}".format(fix_osc))
        self.cart = self.zmat.get_cartesian()
        if overwrite or not os.path.exists(out_file):
            self.cart.to_xyz(out_file)
        else:
            splt = out_file.split(".")
            bname = ".".join(splt[:-1])
            ext = splt[-1]
            count = 1
            while os.path.exists(out_file):
    
                out_file = "{}_{}.{}".format(bname,count,ext)
                count += 1
            self.cart.to_xyz(out_file)
        if view == True:
            self.cart.view(viewer=viewer)
    
    ### NB all @property are "very" private and do not change. zmat._frame["dihedral"] does        
    @property
    def dih_c(self):
        if not hasattr(self, "_dih_c"):
            setattr(self, "_dih_c", self.dih % 360)
        return getattr(self, "_dih_c")
        
    @property
    def dih_mean(self):
        if not hasattr(self, "_dih_mean"):
            setattr(self, "_dih_mean", self.dih.mean(axis=1))
        return getattr(self, "_dih_mean")
    
    @property  # NB this is already backconverted to the (-180,180) range
    def dih_c_mean(self):
        if not hasattr(self, "_dih_c_mean"):
            setattr(self, "_dih_c_mean", conv_d(self.dih_c.mean(axis=1)))  # NB this is already backconverted to the (-180,180) range
        return getattr(self, "_dih_c_mean")
    
    @property
    def dih_std(self):
        if not hasattr(self, "_dih_std"):
            setattr(self, "_dih_std", self.dih.std(axis=1))
        return getattr(self, "_dih_std")
    
    @property
    def dih_c_std(self):
        if not hasattr(self, "_dih_c_std"):
            setattr(self, "_dih_c_std", self.dih_c.std(axis=1))
        return getattr(self, "_dih_c_std")

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

def conv_d(d: int):
    """Convert angles from (0, 360) range
       to (-180, 180) range.

    Parameters
    ----------
    d : int
        Degrees in (-180, 180) range.
    """
    r = d % 360
    return r - (r // 180) * 360 
