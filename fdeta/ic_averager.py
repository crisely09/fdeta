"""Script to make one single file trajectory, one fragment from xyz."""

from typing import Union
import copy as cp
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
from ctypes import cast, py_object
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


rc('text', usetex=True)
vds = {"dihedral": "dih", "dihedrals": "dih", "d": "dih", "dih": "dih",  # Variable Dictionary Singular
         "dih_c": "dih_c",
         "angle": "angle", "angles": "angle", "a": "angle",
         "b": "bond", "bond": "bond", "bonds": "bond"}
vdp = {"dihedral": "dih", "dihedrals": "dih",  "d": "dih", "dih": "dih",
         "angle": "angles", "angles": "angles", "a": "angles",  # Variable Dictionary Plural
         "b": "bonds", "bond": "bonds", "bonds": "bonds"}
vdf = {"dihedral": "dihedral", "d": "dihedral", "dih": "dihedral",  # Variable Dictionary Frame
         "angle": "angles", "angle": "angle", "a": "angle", 
         "b": "bond", "bond": "bond", "bonds": "bond"}

# max-min funcs
nthtolast = lambda x,y: x.index(sorted(x)[-y]) # index of n-th to last 
maxidx = lambda x: x.index(max(x))  # index of max elem, i.e. nthtolast(x,1)
minidx = lambda x: x.index(min(x))  # index of min elem, i.e. nthtolast(x,-1)

def mindidx(l: list):
    """
    Note
    ----
    minimum dual index, i.e. given a list of lists returns index, index to get the minimum
    e.g. mindidx([[10,9],[8,7]]) ==> [1,1]
    
    Parameters
    ----------
    l: list[lists]
        list of lists of int/float
        
    Returns
    -------
    list
        [idx0,idx1]
    
    """
    flat = list(it.chain.from_iterable(l))
    idx = minidx(flat)
    tot = 0
    didx = [0, 0]
    for i in l:
        if tot + len(i) <= idx: 
            didx[0] += 1
            tot += len(i)
        else:
            didx[1] = idx - tot
            break
    return didx

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

def intervals_from_array(arr: Union[np.array, list]):
    """
    Note
    ----
    Given some array, gives a more compact representation using intervals, like printing pages.
    e.g. [1,2,3,4,7,8,9,11,13,14,15,16,17] => "1-4, 7-9, 11, 13-17"
    Useful to mentally process huge arrays of frame numbers
    
    Parameters
    ----------
    arr: list/arr(1D)
    
    Returns
    -------
    str
        the interval-like expression of the array
    """
    n = 0
    strlst = []
    N = len(arr)
    while n < N:
        start = arr[n]
        end = False
        while n < N -1 and arr[n+1] == arr[n] + 1:
            n+=1
            end = arr[n]
        strlst.append("{}-{}".format(start, end) if end else "{}".format(start))
        n += 1
    string = ", ".join(strlst)
    return string
        

def str2zmat(xyz_str: str, c_table: pd.DataFrame, start_index: int =1):
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

def slicestr(slc: Union[slice, list, tuple]):
    """
    Note
    ----
    Gets a string representing the slice object, used for "source" in  ic_avg obtained from slicing
    
    Parameters
    ----------
    slc: slice object
        slice object to obtain the string for
        
    Returns
    -------
    str
        "{start}:{stop}:{step}"
    """
    if type(slc) == slice:
        start = slc.start if slc.start else ""
        stop = slc.stop if slc.stop else ""
        step = slc.step if slc.step else ""
        return "{}:{}:{}".format(start, stop, step)
    elif type(slc) in [list, tuple, np.array]:
        return str(list(slc))

def running_avg(arr: np.array, res: int):
    """
    Parameters
    ----------
    arr: np.arr
        the array to get the running average of
    res: int
        the size of the running interval.
    
    Returns
    -------
    np.arr
        the running average
    """
    return np.convolve(arr, np.ones(res)/res, mode="same")

def ravg_edges(arr: np.array, res: int):
    """
    Note
    ----
    avoids issues at edges by progressively reducing the size of the average
    interval near beginning and end.
    It ensures, for instance, that ravg_edges(np.ones(N), res) => np.ones(N)
    Parameters
    ----------
    arr: np.arr
        the array to get the running average of
    res: int
        the size of the running interval.
    
    Returns
    -------
    np.arr
        the running average
    """
    ravg = running_avg(arr, res)
    offset = res // 2
    for o in range(offset):
        ravg[o] = arr[:o+1].mean()
    offset -= 0 if res%2 == 1 else 1
    for o in range(1, offset+1):
        ravg[-o] = arr[-o:].mean()
    return ravg

def slice_ravg(arr: np.array, left: int, right: int, res: int):
    """
    Note
    ----
    calculates running average of a slice of and array avoiding edge issues.
    It corresponds to slicing the running average of the whole array but avoids 
    calculating it out of the requested slice.
    
    Parameters
    ----------
    arr: np.arr
        the array to get the running average of
    left: int
        start slicing index
    right: int
        end slicing index
    res: int
        the size of the running interval.
    
    Returns
    -------
    np.arr
        the running average in the slice requested
    """
    offset = res // 2
    start = left - offset
    end = right + offset
    ofs_left = offset
    ofs_right = -offset
    flag = False
    if start < 0:
        flag = True
        start = 0
        ofs_left = left
    if start > arr.shape[0] - res:
        start -= res
        ofs_left += res
    if end > arr.shape[0]:
        flag = True
        end = arr.shape[0]
        ofs_right = right - end if end > right else None
    if end < res:
        end +=  res
        ofs_right -= res
    ravg = ravg_edges(arr[start:end], res) if flag else running_avg(arr[start:end], res)
    slc = slice(ofs_left, ofs_right, None)
    return ravg[slc]
    

def plot_2Ddistrib(data: Union[np.ndarray, list], index: list, var: Union[list, str, tuple] = "dihedral",
                   bins: int = 36, labels: list= [],  title: str = "",
                 pos_range: Union[list,bool, type(None)] = None, label: list = []):
    """Plots value occurrence vs value
    Parameters
    ----------
    data: np.array
        the array to slice and plot from. Generally np.array(natoms, nframes)
    index: list or int
        index or list of indexes to plot (data[index,:])
    bins: int
        number of bins for histogram. default is 36
    var: list/tuple/str 
        variable type(s) (dih/angle/bond). used for ranges and for axis labels. default is "dihedral"
    title: str
        title for the plot. default is empty string
    pos_range: list[bool]/bool/None
        whether to plot in the (-180,180) or in the (0,360) range on each axis.
        If boolean, turns to list of twice that value
    
    Returns
    -------
    plt.figure
        the histogram with distribution of data[i,:] for i in index
    """
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    if type(var) == str:
        var = [var, var]
    var = [vds[v] if v in vds.keys() else v for v in var]  
    if type(pos_range) in [bool, type(None)]:
        pos_range = [pos_range, pos_range]
    range_ = [None, None]  
    for n,v in enumerate(var):
        if v == "dih":
            range_[n] = (0, 360) if pos_range[n] else (-180, 180)
        elif v == "angle":
            range_[n] = (0,180)
    if type(data) == list:
        x, y = data[0][index[0], :], data[1][index[1], :]  # two diff var (e.g. bond, angle)
    else:
        x, y = data[index[0], :], data[index[1], :]
    ax.hist2d(x, y, bins=[bins,bins], range=range_)
    if not label:
        ax.set_xlabel("{} {}".format(var[0],index[0]))
        ax.set_ylabel("{} {}".format(var[1],index[1]))
    else:
        ax.set_xlabel("{}".format(label[0]))
        ax.set_ylabel("{}".format(label[1]))
    ax.set_title(title)
    return fig

def plot_distrib(data: np.ndarray, index: Union[list,int],
                 basins: Union[bool, np.ndarray] = False, bins: int = 36,
                 var: str = "dihedral", title: str = "",
                 pos_range: bool = True, label: Union[list, str, int] = [],
                 alpha: float = 0.0, legend: bool = True):  
    """Plots value occurrence vs value
    Parameters
    ----------
    data: np.array
        the array to slice and plot from. Generally np.array(natoms, nframes)
    index: list or int
        index or list of indexes to plot (data[index,:])
    bins: int
        number of bins for histogram. default is 36
    var: str 
        variable type (dih/angle/bond). also used as label for y axis. default is "dihedral"
    title: str
        title for the plot. default is empty string
    pos_range: bool
        whether to plot in the (-180,180) or in the (0,360) range.
        Only applies for var=="dih". 
    label: list/str
        label(s) for the series to plot. length must match that of index
    alpha: float
        transparency. default uses 1.0 for 1 distribution, 0.75 for 2, 0.5 for more
    legend: bool
        whether to plot the legend
        
    Returns
    -------
    plt.figure
        the histogram with distribution of data[i,:] for i in index
    """
    fig = plt.figure(figsize=(20, 10), dpi=150)
    var = vds[var] if var in vds.keys() else var
    ax = fig.add_subplot(111)
    if type(index) in [int, np.int32, np.int64]:
        index = [index]
    if not label:
        label = [str(i) for i in index]
    elif type(label) in [str,int, np.int32, np.int64]:
        label = [str(label)]
    range_ = False
    if var == "dih":
        range_ = (0, 360) if pos_range else (-180, 180)
    elif var == "angle":
        range_ = (0,180)
    if not alpha:
        alpha = 1.0 if len(index) == 1 else 0.75 if len(index) == 2 else 0.5
    if range_:
        if not basins:
            for n,i in enumerate(index):
                ax.hist(data[i, :], bins=bins, range=range_, label=label[n], alpha=alpha)
        else:
            for b in basins:
                for n,i in enumerate(index):
                    ax.hist(data[i, :][b], bins=bins, range=range_, label=label[0], alpha=alpha)
    else:
        for n,i in enumerate(index):
            ax.hist(data[i, :], bins=bins, label=label[n], alpha=alpha)
    ax.set_ylabel("occurrence") 
    if legend:
        ax.legend()
    ax.set_xlabel(var)
    ax.set_title(title)
    return fig


def plot_time_evo(data: np.ndarray, index: Union[list,int], var: str = "dihedral",  
                  basins: Union[bool, np.ndarray] = False, title: str = "",
                  pos_range: bool = True, label: Union[list, str, int] = [],
                  legend: bool = True):
    """Plots value vs time for one or more variables
    Parameters
    ----------
    data: np.array
        the array to slice and plot from. Generally np.array(natoms, nframes)
    index: list or int
        index or list of indexes to plot (data[index,:])
    var: str 
        variable type (dih/angle/bond). also used as label for y axis. default is "dihedral"
    title: str
        title for the plot. default is empty string
    pos_range: bool
        whether to plot in the (-180,180) or in the (0,360) range.
        Only applies for var=="dih". 
    label: list/str
        label(s) for the series to plot. length must match that of index
    legend: bool
        whether to plot the legend
    
    Returns
    -------
    plt.figure
        the value vs t of data[i,:] for i in index
    """
    x = np.arange(data.shape[1])
    var = vds[var] if var in vds.keys() else var
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    if type(index) in [int, np.int32, np.int64]:
        index = [index]
    if not label:
        label = [str(i) for i in index]
    elif type(label) in [str,int, np.int32, np.int64]:
        label = [label]
    if basins:
        for b in basins:
            for n,i in enumerate(index):
                ax.scatter(x[b], data[i, :][b], label=label[0])
    else:
        for n,i in enumerate(index):
            ax.scatter(x, data[i, :], label=label[n])
    ax.set_ylabel(var)  
    ax.set_xlabel("frame")
    if legend:
        ax.legend()
    ax.set_title(title)
    if var == "dih":
        if not pos_range:
            ax.set_ylim(-180, 180)
        else:
            ax.set_ylim(0, 360)
    elif var == "angle":
        ax.set_ylim(0, 180)
    return fig

def grouplist_to_indexes(group_list: list, ic: bool = True, cart: bool = False, numbering: str = ""):
    """
    Note
    ----
    Gets all indexes in a grouplist
    
    Parameters
    ----------
    group:list: list
        the group_list
    ic: bool
        whether to get the ic-indexes
    cart: bool 
        whether to get the cartesian indexes
    numbering: str
        indexes to return
    
    Returns
    -------
    list/(list,list)
        list if only one type, (ic, cart) if both
    """
    if numbering:
        if "ic" in numbering or "internal" in numbering:
            ic = True
        else: 
            ic = False
        if "cart" in numbering:
            cart = True
    if ic:
        ic_list = list(it.chain.from_iterable([gr.arr for gr in group_list]))
    if cart:
        cart_list = list(it.chain.from_iterable([gr.cart for gr in group_list]))
    to_return = (ic_list, cart_list) if ic and cart else ic_list if ic else cart_list
    return to_return

class Group:
    """
    Object for a set of variables, most commonly dihedrals, corresponding to the same chemical group.
    For instance 3 H in a methyl, or 2 H in an amino group, or 1 H in a hydroxy
    """
    def __init__(self, arr: Union[np.array, list, tuple], avg_id: int, var: str = "dih"):
        """
        arr: np.array, list, tuple
            indexes of the angles in ic-numbering
        avg_id: int
            id of the averager the group refers to
        var: str
            type of variable. default is "dih"
            uses vds for dihedral/dih, angles/angle, etc...
        """
        if type(arr) in [int, np.int32, np.int64]:
            arr = np.array([arr])
        if type(arr) in [list,tuple]:
            arr = np.array(arr)
        self.arr = arr
        self.avg_id = avg_id  # id(ic_avg), works as pointer to the averager
        self.var = vds[var] if var in vds.keys() else var
        
    def __repr__(self):
        """
        Note
        ----
        When calling group, it returns a string with ic-numbering and, if available, cartesian counting
        """
        if hasattr(self, "_cart"):
            return "{} = IC: {}, Cart: {}".format(self.var, self.arr.tolist(), self.cart.tolist())
        else:
            return "{} = IC: {}".format(self.var, self.arr.tolist())
        
    def __str__(self):
        """
        Note
        ----
        When converting group to a string (e.g. printing), it returns a string with ic-numbering and, if available, cartesian counting
        """
        if hasattr(self, "_cart"):
            return "{} = IC: {}, Cart: {}".format(self.var, self.arr.tolist(), self.cart.tolist())
        else:
            return "{} = IC: {}".format(self.var, self.arr.tolist())
        
    @staticmethod
    def from_cart(cart: np.array, avg_id: int, var = "dih"):
        """
        Note
        ----
        Obtains a group from indexes in cartesian counting
        
        Parameters
        ----------
        cart: np.arr, list, tuple
            indexes in cartesian numbering
        avg_id: int
            id of the ic_avg the group refers to
        """
        if type(cart) in [list,tuple]:
            cart = np.array(cart)
        c_table = cast(avg_id, py_object).value.c_table
        arr = np.array([c_table.index.get_loc(i) for i in cart])
        gr = Group(arr, avg_id, var=var)
        gr._cart = cart
        return gr
    
    @property
    def cart(self):
        """
        Note
        ----
        Some properties are private attributes with property decorator.
        Reduces the risk of the user messing with it, but allows skilled users to do so.
        """
        if not hasattr(self, "_cart"):
            c_table = cast(self.avg_id, py_object).value.c_table
            setattr(self, "_cart", np.array(c_table.index[self.arr]))
        return getattr(self, "_cart")
    
    def get_basins(self, min_centers: int = 2, max_centers: int = 3, overwrite: bool = False):
        """
        Note
        ----
        Obtains basins for all variables within the group
        
        Parameters
        ----------
        min_centers: int
            minimum number of centers, default i 2
        max_centers: int
            maximum number of centers, default is 3
        overwrite: bool
            whether present basins should be overwritten or not, default is False
        """
        if overwrite or not hasattr(self,"_basins"):
            print("calculating basins with clustering")
            avg = cast(self.avg_id, py_object).value
            basins, centers = [], []
            for i in self.arr:
                b, c = avg.find_clusters(var=self.var, index=i, min_centers=min_centers, max_centers=max_centers)
                basins.append(b), centers.append(c)
            self._basins, self._centers = basins, centers
            
    def basins_from_time_domains(self, overwrite: bool = False, thresh_chunk: Union[int, float] = 25,
                                 thresh_frame : Union[int, float] = 5, max_counter :  int = 3,
                                 res_chunks: int = 15, res_ravg: int = 10,
                                 recalculate: bool = False, thresh_centers: Union[int, float] = 4,
                                 check_thresh: Union[int, float] = 10):
        """
        Note
        ----
        Gets basins from time domains. Uses threshold for centers of time domains
        to decide if they pertain to the same basin.
        
        Parameters
        ----------
        overwrite: bool
            whether present basins should be overwritten or not, default is False
        thresh_chunk: int/float
            threshold to detect approximate position of a jump via chunk displacement
        thresh_frame: int/float
            threshold to detect exact position of a jump via running average displacement
        max_counter: int
            number or consecutive values of running average above thresh_frame necessary
            for a jump
        res_chunks: int
            size of the chunks for the chunk displacement
        res_ravg: int
            resolution/interval for the running average. 
            res_ravg > res_chunks raises error
        recalculate: bool
            whether the chunk displacement should be recalculated
        thresh_centers: int/float
            threshold for centers to the closest one
        check_thresh: int/float
            threshold for min-max center of domains grouped into a basin
            
        Sets
        ----
        self._basins, self._centers
            lists of frames and centers
        """
        if overwrite or not hasattr(self,"_basins"):
            print("calculating basins from time domains")
            avg = cast(self.avg_id, py_object).value
            domains = avg.get_time_domains(var=self.var, index=self.arr, thresh_chunk=thresh_chunk,
                                           thresh_frame=thresh_frame, max_counter=max_counter,
                                           res_chunks=res_chunks,  res_ravg=res_ravg, recalculate=recalculate)
            bs, cs = avg.time_domains_to_basins(domains, var=self.var, thresh=thresh_centers,
                                                check_thresh=check_thresh)
            basins, centers = [bs[i] for i in self.arr], [cs[i] for i in self.arr]
            self._basins, self._centers = basins, centers        
    
    def get_frames(self, basin: Union[int, str] = 1):
        """
        Note
        ----
        Gets frame numbers for a specific basin
        
        Parameters
        ----------
        basin: int/str
            int for n-th basin by size, "res" for the one that minimises residuals
        
        Returns
        -------
        list
            the frames of the desired basin for the selected atom
        """
        if type(basin) in [int, np.int32, np.int64]:
            return self.basins[self.selected][nthtolast(self.weights[self.selected], basin)]  
        elif basin == "res":
            self.select_res()
            return self.basins[self.selected][mindidx(self.res)[1]]  # center with lowest residuals
        
    @property
    def basins(self):
        """
        Note
        ----
        Some properties are private attributes with property decorator.
        Reduces the risk of the user messing with it, but allows skilled users to do so.
        """
        self.get_basins()
        return self._basins
    
    def set_basins(self, basins: list):
        """
        Note
        ----
        Sets basins to the provided lists, automatically updates centers.
        Useful if one wants to tinker with basins found with other functions (e.g. join two basins)
        
        Parameters
        ----------
        basins: list
            list of frames for each frame for each atom.
            nb. the size is not checked
        
        Sets
        ----
        self._basins, self._centers
        """
        self._basins = basins
        avg = cast(self.avg_id, py_object).value
        var_traj = getattr(avg, vdp[self.var]) if self.var in ["bond", "angle"] else getattr(avg, self.var)
        centers = []
        for n,i in enumerate(self.arr):
            centers.append([var_traj[i][b].mean() for b in basins[n]])
        self._centers = centers
        print("set centers")
        
    @property
    def centers(self):
        """
        Note
        ----
        Some properties are private attributes with property decorator.
        Reduces the risk of the user messing with it, but allows skilled users to do so.
        """
        self.get_basins()
        return self._centers
    
    @property
    def selected(self):
        """
        Note
        ----
        Some properties are private attributes with property decorator.
        Reduces the risk of the user messing with it, but allows skilled users to do so.
        
        This atom will be used to pick a basin, then the other atoms are assigned in such a way 
        that the difference is the same as the average one in the basin.
        e.g. for methyl, H1-H2 =~ 120°
        """
        if hasattr(self, "_selected"):
            return self._selected
        else:
            if len(self.basins) == 1:
                self._selected = 0
            else:
                print("the \"selected\" atom has not been set yet")
                return None
        
    @selected.setter
    def selected(self, idx: int):
        """
        Note
        ----
        Setter for self._selected
        
        Parameters
        ----------
        idx: int
            the 
        """
        if hasattr(self, "_selected") and self._selected != idx and hasattr(self,"_avg_values"):
            del self._avg_values
        self._selected = idx
        
    @property
    def weights(self):
        """
        Note
        ----
        Some properties are private attributes with property decorator.
        Reduces the risk of the user messing with it, but allows skilled users to do so.
        
        weight of each basin, i.e. number of frames in each basin
        """
        if not hasattr(self,"_weights"):
            weights = []
            for basins in self.basins:
                weights.append([len(basin) for basin in basins])
            self._weights = weights
        return self._weights
    
    @property
    def prominence(self):
        """
        Note
        ----
        weight difference between the first and second most common basin
        """
        if not hasattr(self,"_prominence"):
            self._prominence = [sorted(i)[-1] - sorted(i)[-2] for i in self.weights]
        return self._prominence
    
    def select(self, criterion: Union[str, callable]):  
        """
        Parameters
        ----------
        criterion: str/callable
            how to select. "res" calls self.select_res()
            otherwise same as select_prominence
        """
        if criterion == "res":
            self.select_res()
        elif criterion in ["min", "minimum", "least", "less", min]:
            self.select_least_prominent()
        elif criterion in ["max", "maximum", "most", "more", max]:
            self.select_most_prominent()
            
    def select_most_prominent(self):
        """
        Note
        ----
        Selects the angle with largest prominence
        """
        if len(self.basins) == 1:
            self._selected = 0
        else:
            self._selected = self.prominence.index(max(self.prominence))
            
    def select_least_prominent(self):
        """
        Note
        ----
        Selects the angle with smallest prominence
        """
        if len(self.basins) == 1:
            self._selected = 0
        else:
            self._selected = self.prominence.index(min(self.prominence))
            
    def select_prominence(self, minormax: Union[str, callable]):
        """
        Note
        ----
        Selects the angle with largest/smallest prominence
        
        Parameters
        ----------
        minormax: str/func
            min/max, "min"/"max", "minimum"/"maximum", "lest"/"most", "less"/"more"
        """
        if minormax in ["min", "minimum", "least", "less", min]:
            self.select_least_prominent()
        elif minormax in ["max", "maximum", "most", "more", max]:
            self.select_most_prominent()
        else:
            raise ValueError("Use min/minimum/least/less or max/maximum/most/more")
            
    def res_analysis(self):  
        """
        Note
        Analysis of "residuals". 
        i.e. looks for the combination of basins which minimises difference between 
        angle differences for the basins centers and average angle differences over the trajectory
        """
        if self.var != "dih":
            raise ValueError("residuals are only for dihedrals")
        if len(self.centers) == 1:
            res = [[0 for c in self.centers[0]]]
        else:
            avg = cast(self.avg_id, py_object).value
            res = [[0 for i in j] for j in self.centers]
            for na,angle in enumerate(self.centers):  # over angles in group
                others = [(m,i) for m,i in enumerate(self.arr) if m != na]  # (numbering in group, numbering in ic) for other_angle != angle
                mdiffs = [(conv_d(avg.dih[self.arr[na],:] - avg.dih[[i[1]],:])).mean() for i in others]  # mean differences other-angle over the trajectory
                for nc,c_ in enumerate(self.centers[na]):  # centers for that angle
                    for no,o in enumerate(others):
                        expected = conv_d(c_ - mdiffs[no])   # expected value based on avg diff
                        res[na][nc] += min([abs(expected - i) for other in others for i in self.centers[other[0]]])  # abs diff between expected and center
        self._res = res
            
    @property
    def res(self):
        """
        Note
        ----
        Sum of residuals for each center
        """
        if not hasattr(self, "_res"):
            self.res_analysis()
        return self._res
    
    def select_res(self):
        """
        Note
        ----
        Selects atom with the lowest residuals
        """
        self.selected = mindidx(self.res)[0]
    
    def time_evo_sep(self, title: str = "", centers: bool = False, basins: bool = False,
                     legend: bool = True, displacement: bool = False, ravg: bool = False,
                     res_displ: int = 15, res_ravg: int = 10, recalculate: bool = False):
        """
        Note
        ----
        Plots time evolution of each variable in group, each in a separate picture
        
        Parameters
        ----------
        title: str
            title for the plot. default is empty string
        centers: bool
            whether the centers should be plotted (dashed lines)
        basins: bool
            whether the basins should be plotted with different colors
        displacement: bool
            whether the chunk displacement should be plotted
        ravg: bool
            whether the running average should be plotted
        res_displ: int
            the size of the chunks for the displacement
        res_ravg: int
            the time interval for the running average
        recalculate: bool
            whether running average and/or displacement should be recalculated
            (use if changing res)
        Returns
        -------
        plt.figures
            the value vs t of data[i,:] for i in index
        """
        avg = cast(self.avg_id, py_object).value
        to_return = []
        for n,a in enumerate(self.arr):
            fig = avg.plot_time_evo(index=a, basins=self.basins[n] if basins else False,
                                    var=self.var, title=title, label=self.cart[n],
                                    legend=legend, displacement=displacement,
                                    ravg=ravg, res_displ=res_displ, res_ravg=res_ravg,
                                    recalculate=recalculate)
            if centers:
                ax = fig.axes[0]
                using_c = a in avg.use_c
                ax.hlines([i%360 for i in self.centers[n]] if using_c else [conv_d(i) for i in self.centers[n]], *ax.get_xlim(), linestyle="--", alpha=0.5)
            to_return.append(fig)
        return to_return
    
    def time_evo(self, title: str = "", displacement: bool = False, ravg: bool = False,
                 res_displ: int = 15, res_ravg: int = 10, recalculate: bool = False):
        """
        Note
        ----
        Plots time evolution of each variable in group, all in the same picture
        
        Parameters
        ----------
        title: str
            title for the plot. default is empty string
        displacement: bool
            whether the chunk displacement should be plotted
        ravg: bool
            whether the running average should be plotted
        res_displ: int
            the size of the chunks for the displacement
        res_ravg: int
            the time interval for the running average
        recalculate: bool
            whether running average and/or displacement should be recalculated
            (use if changing res)
        Returns
        -------
        plt.figure
            the value vs t of data[i,:] for i in index
        """
        avg = cast(self.avg_id, py_object).value
        fig = avg.plot_time_evo(index=self.arr.tolist(), var=self.var, title=title,
                                label=list(self.cart), displacement=displacement,
                                ravg=ravg, res_displ=res_displ, res_ravg=res_ravg,
                                recalculate=recalculate)
        return fig
    
    def distrib_sep(self, bins: int = 36, title: str = "", alpha: float = 0.0, centers: bool = False, basins: bool = False, legend: bool = True):
        """
        Note
        ----
        Plots time evolution of each variable in group, each in a separate picture
        
        Parameters
        ----------
        bins: int
            the number of bins
        title: str
            title for the plot. default is empty string
        alpha: float
            transparency. default uses 1.0 for 1 distribution, 0.75 for 2, 0.5 for more
        
        Returns
        -------
        plt.figures
            the value vs t of data[i,:] for i in index
        """
        avg = cast(self.avg_id, py_object).value
        to_return = []
        for n,a in enumerate(self.arr):
            fig = avg.plot_distrib(a, basins=self.basins[n] if basins else False, bins=bins, var=self.var, title=title, label=self.cart[n], alpha=alpha, legend=legend)
            if centers:
                ax = fig.axes[0]
                using_c = a in avg.use_c
                ax.vlines([i%360 for i in self.centers[n]] if using_c else [conv_d(i) for i in self.centers[n]], *ax.get_ylim(), linestyle="--", alpha=0.5)
            to_return.append(fig)
        return to_return
    
    def distrib(self, bins: int = 36, title: str = "", alpha: float = 0.0):  
        """
        Note
        ----
        Plots time evolution of each variable in group, all in the same picture
        
        Parameters
        ----------
        bins: int
            the number of bins
        title: str
            title for the plot. default is empty string
        alpha: float
            transparency. default uses 1.0 for 1 distribution, 0.75 for 2, 0.5 for more
        
        Returns
        -------
        plt.figure
            the value vs t of data[i,:] for i in index
        """
        avg = cast(self.avg_id, py_object).value
        return avg.plot_distrib(self.arr.tolist(), bins=bins, var=self.var, title=title, label=list(self.cart), alpha=alpha)
            
    def get_avg_values(self, basin: int = 1, overwrite: bool = False, return_frames : bool  = False):  
        """
        Note
        ----
        Gets average values for the desired basin
        
        Parameters
        ----------
        basin: int/"res"
            if basin=n the n-th to last basin per weight is used (1=first basin, 2=second basin...)
            "res" uses the basin that minimises residual sums
        overwrite: bool
            whether present basins should be overwritten or not, default is False
            
        Returns
        -------
        list if not return frames else tuple(list, np.ndarray) the average values
        """
        if not hasattr(self,"_avg_values") or overwrite:
            vals = []
            if basin == "res":
                self.select_res()
                b = mindidx(self.res)[1]
            elif type(basin) in [int, np.int32, np.int64]:
                b = nthtolast(self.weights[self.selected], basin)
            else:
                raise ValueError("unrecognised value for \"basin\": it must be an implemented method to select the basin")
            self.frames_used_idx = (cp.copy(self.selected), cp.copy(b))  # dual index to get basin_frames. NB: Copy!
            basin_frames = self.basins[self.selected][b]
            tosub = self.centers[self.selected][b]
            avg = cast(self.avg_id, py_object).value
            for n, basins in enumerate(self.basins):
                if n == self.selected:
                    vals.append(tosub)
                else:
                    if self.var == "dih":
                        diff = avg.dih[self.arr[self.selected],:][basin_frames] - avg.dih[self.arr[n],:][basin_frames]
                        if diff.std() > conv_d(diff).std():  # 
                            diff = conv_d(diff)
                        diff = conv_d(diff.mean())
                        vals.append(conv_d(tosub - diff))
                    else:
                        vals.append(avg.dih[self.arr[n],:][basin_frames].mean())
            self._avg_values = vals
        to_return = (self._avg_values, self.basins[self.frames_used_idx[0]][self.frames_used_idx[1]]) if return_frames else self._avg_values
        return to_return
    
def intersect_lists(ll: list):  # tested to be faster than other possible methods
    """
    Note
    ----
    Given a list of lists, returns the list of elements present in all of them.
    
    Parameters
    ----------
    ll: list[lists]
        list of lists (generally of frame numbers)
    
    Returns
    -------
    list
        the intersection
    """
    intersec = set(ll[0])
    for l in ll[1:]:
        intersec = intersec & set(l)
    return list(intersec)

def get_frames(gr: Group, basin: Union[int, str] = 1):
    """
    Note
    ----
    Gets frame numbers for a specific basin
    
    Parameters
    ----------
    gr: Group
        the group to get the frames from
    basin: int/str
        int for n-th basin by size, "res" for the one that minimises residuals
    
    Returns
    -------
    list
        the frames of the desired basin for the selected atom
    """
    if type(basin) in [int, np.int32, np.int64]:
        return gr.basins[gr.selected][nthtolast(gr.weights[gr.selected], basin)]  
    elif basin == "res":
        gr.select_res()
        return gr.basins[gr.selected][mindidx(gr.res)[1]]  # center with lowest residuals
        
def intersect_groups(grouplist: list, basin: Union[int, str] = 1):  
    """
    Note
    ----
    Given a list of groups, returns the frames of the common basin,
    i.e. the intersection of the desired basins for the selected atoms of each group
    
    Parameters
    ----------
    basin: int/"res"/
        if basin=n the n-th to last basin per weight is used (1=first basin, 2=second basin...)
        "res" uses the basin that minimises residual sums
    
    Returns
    -------
    list
        The intersection of these basins (list of framenumbers)
    """
    if type(basin) in [int, np.int32, np.int64]:
        gt_frms = lambda gr: gr.basins[gr.selected][nthtolast(gr.weights[gr.selected], basin)]  
    elif basin == "res":
        def gt_frms(gr):
            gr.select_res()
            return gr.basins[gr.selected][mindidx(gr.res)[1]]  # center with lowest residuals
    ll = [gt_frms(gr) for gr in grouplist]
    return intersect_lists(ll)

def agreement_lists(ll: list):
    """
    Parameters
    ----------
    ll: list[lists]
        
    Returns
    -------
    float
        the percentage of frames which appear in all basins
    """
    return len(intersect_lists(ll))/max([len(l) for l in ll])

def agreement_groups(grouplist: list, basin: int = 1):  
    """
    Parameters
    ----------
    ll: list[groups]
        
    Returns
    -------
    float
        the percentage of frames which appear in all selected basins
    """
    if type(basin) in [int, np.int32, np.int64]:
        gt_frms = lambda gr: gr.basins[gr.selected][nthtolast(gr.weights[gr.selected], basin)]  
    elif basin == "res":
        def gt_frms(gr):
            gr.select_res
            return gr.centers[gr.selected][mindidx(gr.res)[1]]
    ll = [gt_frms(gr) for gr in grouplist]
    return agreement_lists(ll)

class Ic_averager:
    """Object for averaging of molecule A in internal coordinates.
    Allows different methods to detect and fix issues such as quasilinear angles and rotating groups.
    """

    def __init__(self, source: str, bonds: np.ndarray = np.array([]), angles: np.ndarray = np.array([]),
                 dih: np.ndarray = np.array([]), zmat1: Union[cc.Zmat, None] = None,  
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
        self.source = cp.copy(source)
        self.bonds = bonds.copy()
        self.angles = angles.copy()
        self.dih = dih.copy()
        self.zmat1 = zmat1.copy()
        self.c_table = c_table.copy()
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
    
    def add_pseudo(self, name: str, arr: np.ndarray, overwrite: bool = False):  
        """
        Note
        ----
        Simply sets self.name = arr but checks there are nframe values
        
        Parameters
        ----------
        name: str
            desired name for the pseudo coordinate
        arr: np.array
            the array of pseudo coordinate values
        overwrite: bool
            whether we should overwrite values or raise an error if the pseudocoordinate exists already
            
        Sets
        ----
        self.name
        """
        nframes = arr.shape[0] if len(arr.shape) == 1 else arr.shape[1]
        if nframes != self.nframes:
            raise ValueError("The number of frames does not match!!")
        if not overwrite and hasattr(self, name):
            raise ValueError("This pseudo-coordinate seems to be already there! Check, and use \"overwrite=True\" in case")
        setattr(self, name, arr)
        
    def pseudos_from(self, var: str = "", overwrite: bool = False, **kwargs):  
        """
        Note
        ----
        if the pseudocoordinate only depends on 1 type of coordinates (e.g. only bonds), use var="bonds",
        and specify the function to obtain the pseudo coordinate from the bonds of a frame.
        If the pseudocoordinate depends on more types of coordinates, leave var="" (default) and provide a function
        which accepts **kwargs and uses kwargs["bonds"], kwargs["angles"], kwargs["dih"]
        
        Parameters
        ----------
        var: str
            the variable type to use to obtain pseudo, if only one
            otherwise use default, which is ""
        
        overwrite: bool
            whether we should overwrite values or raise an error if the pseudocoordinate exists already
        **kwargs: dict
            "pseudo_name1": function1
            "pseudo_name2": function2
        
        Sets
        ----
        self.pseudo = np.array(k, nframes) 
            k depends on your function, most commonly 1
        """
        if var:  # we can use apply_along_axis
            var = vdp[var]
            arr = getattr(self, var)
            for coord, func in kwargs.items():
                if not overwrite and hasattr(self, coord):
                    raise ValueError("This pseudo-coordinate seems to be already there! Check, and use \"overwrite=True\" in case")
                pseudo = np.apply_along_axis(func, 0, arr)
                if len(pseudo.shape) == 1:
                    pseudo = pseudo.reshape(1,-1)
                setattr(self, coord, pseudo)
        else:
            for coord, func in kwargs.items():
                if not overwrite and hasattr(self, coord):
                    raise ValueError("This pseudo-coordinate seems to be already there! Check, and use \"overwrite=True\" in case")
                pseudo = np.zeros(self.nframes)
                for f in range(self.nframes):
                    bonds, angles, dih = self.bonds[:, f], self.angles[:, f], self.dih[:,f]  # test 2d pseudo
                    pseudo[f] = func(bonds=bonds, angles=angles, dih=dih)
                setattr(self, coord, pseudo)
    
    def get_displacement(self, var: str = "dih", index: Union[list, np.ndarray, int, str] = [], res: int = 15):
        """
        Parameters
        ----------
        var: str
            the variable to use (angle, bond, dih, dih_c, some pseudo)
        index: list/array, int, sr
            the indexes to get the displacement for. use "all" to get all natoms
        res: int
            the resolution. e.g. 10 averages the values every 10 frames
        
        Sets
        ----
        self.var_d: dict
            dictionary of {idx: [array of the displacement] for idx in indexes}
        """
        if index in ([], np.array([])):
            raise ValueError("You must specify some angles! (in ic_numbering)")
        elif type(index) in [int, np.int32, np.int64]:
            index= [index]
        elif index is "all":  # not == because on np.arrays
            index = np.arange(self.natoms)
        var = vds[var] if var in vds.keys() else var
        if "dih" in var:
            indexes = [("dih_c", [i for i in index if i in self.use_c]), ("dih", [i for i in index if i not in self.use_c])]
        else:
            indexes = [(var, index)]
        for j in indexes:
#            if len(j) != 2:
#                continue
            v, idx = j
            if not hasattr(self, "{}_d".format(v)):
                setattr(self, "{}_d".format(v), {})
            if not hasattr(self, "{}_s".format(v)):
                setattr(self, "{}_s".format(v), {})
            orig = getattr(self, vdp[v]) if v in ["bond", "angle"] else getattr(self, v)
            d = {}
            c = {}
            if res == 1:
                for i in idx:
                    d[i] = (np.diff(orig[i]), 1)
            else:
                for i in idx:
                    arr = np.zeros(int(orig[i].size/res))
                    for k in range(arr.size):
                        arr[k] += orig[i, int(k*res):int((k+1)*res)].mean()
                    c[i] = arr
                    d[i] = (np.diff(arr), res)
            getattr(self, "{}_d".format(v)).update(d)
            getattr(self, "{}_s".format(v)).update(c)
    
    def displacement_analysis(self, var: str = "dih", index: Union[list, np.ndarray, int, str] = "all",
                              thresh_chunk: Union[int, float] = 25, res_chunks: int = 15, recalculate: bool = False):
        """
        Note
        ----
        Obtains chunk displacement, detects estimated position of jumps above threshold
        
        Parameters
        ----------
        var: str
            variable. default is "dih"
        index: list, array, "all"
            the index(es) to analyse. use all for np.arange(natoms)
        thresh_chunk: int/float
            threshold for displacement above which a jump is detected
        res_chunk: int
            size of the chunks used for the displacement
        recalculate: bool
            whether the displacement should be recalculated
            
        Returns
        -------
        dict
            {idx: [estimated jumps positions]}
        """
        if index in ([], np.array([])):
            raise ValueError("You must specify some indexes! (in ic_numbering)")
        elif type(index) in [int, np.int32, np.int64]:
            index= [index]
        elif index is "all":  # not == because on np.arrays
            index = np.arange(self.natoms)
        var = vds[var] if var in vds.keys() else var
        if "dih" in var:
            indexes = [("dih_c", [i for i in index if i in self.use_c]), ("dih", [i for i in index if i not in self.use_c])]
        else:
            indexes = [(var, index)]
        est_j = {}  # estimated jump positions
        for z in indexes:
            v, idx = z
            if not idx:
                continue
            if recalculate or not hasattr(self, "{}_d".format(vds[v])):
                self.get_displacement(var=v, index=idx, res=res_chunks)
                displ_d = getattr(self, "{}_d".format(vds[v]))
            else:
                displ_d = getattr(self, "{}_d".format(vds[v]))
                missing = [i for i in idx if i not in displ_d.keys()]
                if missing:
                    self.get_displacement(var=v, index=missing, res=res_chunks)
            for i in idx:
                displ, res = displ_d[i]
                hits = np.where(abs(displ) > thresh_chunk)[0]
                jumps =[]
                last_present = True if displ.shape[0] - 1 in hits else False  # avoids indexing h+1 if that would cause IndexError
                if last_present:
                    hits = hits[:-1]
                for h in hits:  # hits to jumps
                    if h+1 in hits:
                        if displ[h]*displ[h+1] < 0:  # big change of sign => two jumps!
                           jumps.append(h) # add current (and will add next)
                        continue  # if not big jump only add second hit
                    if abs(displ[h+1]) < 0.9*thresh_chunk:
                        jumps.append(h)  # simplest case
                    elif displ[h]*displ[h+1] > 0:  # no big change of sign
                        jumps.append(h+1)  # add only value next to hit
                    else:
                        jumps.extend([h, h+1])  # big change of sign, add hit and next value
                if last_present:
                    jumps.append(displ.shape[0] - 1)
                est_j[i] = [(j+1)*res for j in jumps]
        return est_j
                
    def get_ravg(self, var: str = "dih", index: Union[list, np.ndarray, int, str] = [],
                 res: int = 15, fix_edges: bool = True):
        """
        Note
        ----
        Gets running average for a specified variable and index and saves it
        
        Parameters
        ----------
        var: str
            variable. default is "dih"
        index: list, array, "all"
            the index(es) to analyse. use all for np.arange(natoms)
        res: int
            the time interval for the running average
        fix_edges: bool
            whether boundary issues should be fixed
            
        Sets
        ----
        self.var_m[idx] = (array of ravg, res)
        """
        if index in ([], np.array([])):
            raise ValueError("You must specify some angles! (in ic_numbering)")
        elif type(index) in [int, np.int32, np.int64]:
            index= [index]
        elif index is "all":  # not == because on np.arrays
            index = np.arange(self.natoms)
        var = vds[var] if var in vds.keys() else var
        if "dih" in var:
            indexes = [("dih_c", [i for i in index if i in self.use_c]), ("dih", [i for i in index if i not in self.use_c])]
        else:
            indexes = [(var, index)]
        for j in indexes:
#            if len(j) != 2:
#                continue
            v, idx = j
            if not hasattr(self, "{}_m".format(v)):
                setattr(self, "{}_m".format(v), {})
            orig = getattr(self, vdp[v]) if v in ["bond", "angle"] else getattr(self, v)
            d = {}
            if res == 1:
                for i in idx:
                    d[i] = (orig, 1)
            else:
                for i in idx:
                    ravg = ravg_edges(orig[i], res) if fix_edges else running_avg(orig[i], res)
                    d[i] = (ravg, res)
            getattr(self, "{}_m".format(v)).update(d)
            
    def get_time_domains(self, var: str = "dih", index: Union[list, np.ndarray, int, str] = [],
                              thresh_chunk: Union[int, float] = 25, thresh_frame : Union[int, float] = 5,
                              max_counter :  int = 3, res_chunks: int = 15, res_ravg: int = 10,
                              recalculate: bool = False):
        """
        Note
        ----
        detects precise position of jumps in trajectory yielding time domains.
        
        Parameters
        ----------
        var: str
            variable. default is "dih"
        index: list, array, "all"
            the index(es) to analyse. use all for np.arange(natoms)
        thresh_chunk: int/float
            threshold for displacement above which a jump is detected
        thresh_frame: int/float
            increase in ravg above which we consider a jump (if happens max_counter times in a row)
        max_counter: int
            how many increases of ravg of more than thresh_frame need to happen to constitute a jump
            if fewer, it is considered fluctuation
        res_chunk: int
            size of the chunks used for the displacement
        res_ravg: int
            time interval for the running average. No more than res_chunk
        recalculate: bool
            whether the displacement should be recalculated
            
        """
        if index in ([], np.array([])):
            raise ValueError("You must specify some indexes! (in ic_numbering)")
        elif type(index) in [int, np.int32, np.int64]:
            index= [index]
        elif index is "all":  # not == because on np.arrays
            index = np.arange(self.natoms)
        var = vds[var] if var in vds.keys() else var
        if "dih" in var:
            indexes = [("dih_c", [i for i in index if i in self.use_c]), ("dih", [i for i in index if i not in self.use_c])]
        else:
            indexes = [(var, index)]
        beginnings, ends = {}, {}
        for z in indexes:
            v, idx = z
            if not idx:
                continue
            if recalculate or not hasattr(self, "{}_d".format(vds[v])):
                self.get_displacement(var=v, index=idx, res=res_chunks)
                displ_d = getattr(self, "{}_d".format(vds[v]))
            else:
                displ_d = getattr(self, "{}_d".format(vds[v]))
                missing = [i for i in idx if i not in displ_d.keys()]
                if missing:
                    self.get_displacement(var=v, index=missing, res=res_chunks)
            orig = getattr(self, vdp[v]) if v in ["bond", "angle"] else getattr(self, v)
            for i in idx:
                displ, res = displ_d[i]
                if res < res_ravg:
                    raise ValueError("cannot have res_ravg larger than the chunk resolution of your displacement")
                hits = np.where(abs(displ) > thresh_chunk)[0]
                jumps =[]
                for h in hits:  # hits to jumps
                    if h+1 in hits:
                        if displ[h]*displ[h+1] < 0:  # big change of sign => two jumps!
                           jumps.append(h) # add current (and will add next)
                        continue  # if not big jump only add second hit
                    if abs(displ[h+1]) < 0.9*thresh_chunk:
                        jumps.append(h)  # simplest case
                    elif displ[h]*displ[h+1] > 0:  # no big change of sign
                        jumps.append(h+1)  # add only value next to hit
                    else:
                        jumps.extend([h, h+1])  # big change of sign, add hit and next value
                beginnings[i], ends[i] = [], []
                for j in jumps:
                    left = j*res - 1  # +j from displ to chunk_mean, -j to avoid the jump, -1 for ravg_d
                    right = (j+2)*res + 1  # +j from displ to chunk_mean, +j to avoid the jump, +1 for ravg_d
                    left = left if left >= 0 else 0
                    right = right if right <= self.nframes else self.nframes 
                    ravg = slice_ravg(orig[i], left, right, res_ravg)  # same as running_average(orig[i])[left, right]
                    ravg_d = np.diff(ravg)
                    delims = [0, ravg_d.shape[0]-1]
                    lims_list = [ends[i], beginnings[i]]
                    incr = [1, -1]
                    for n in range(2):
                        cntr = 0
                        delim = delims[n]
                        lims= lims_list[n]
                        cycles = 0
                        while 0 <= delim < ravg_d.shape[0] if n == 0 else 0 < delim <= ravg_d.shape[0] :  # never go in infite negative items loop!
                            cycles += 1
                            condition = ravg_d[delim] > thresh_frame if displ[j] > 0 else ravg_d[delim] < -thresh_frame
                            if not condition:
                                cntr = 0
                            if condition:
                                cntr += 1
                                if cntr == max_counter:
                                    delim -= incr[n]*(max_counter-1)  # beginning of real descent/ascent (end flat region)
                                    break
                            delim += incr[n]
                        lims.append(left + delim)
        domains = {}
        for i in beginnings.keys():
            bs = [0] + beginnings[i]
            es = ends[i] + [self.nframes]
            edges = list(zip(bs, es))
            domains[i] = [np.arange(*e) for e in edges] 
        return domains
    
    def time_domains_to_basins(self,  domains: Union[dict, list], var: str = "dih",
                               index: Union[list, np.ndarray, int, str] = [],
                               thresh: Union[int, float] = 3, check_thresh: Union[int, float] = 10):
        """
        Note
        ----
        Joins domains with similar centers
        
        Parameters
        ----------
        domains: dict, list
            dictionary of idx = domains or list if only one idx
        var: str
            variable. default is "dih"
        index: list, array, "all"
            the index(es) to analyse. use all for np.arange(natoms)
        thresh: int/float
            max distance of centers one to the closest to be put in the same basin
        check_thresh: int/float
            threshold for min-max of domain centers within a basin
            e.g. if centers = [180,181,182,...,195] they are all within 1° of the closest,
            but if check_thresh < 15 an error will be raised
        
        Returns
        -------
        tuple(list, list)
            basins, centers
        """
        if index in ([], np.array([])):
           if type(domains) == dict:
               index = domains.keys()
           else:
               raise ValueError("You must specify some indexes! (in ic_numbering)")
        elif type(index) in [int, np.int32, np.int64]:
            index= [index]
        elif index is "all":  # not == because on np.arrays
            index = np.arange(self.natoms)
        if type(domains) == list:
            if len(index) == 1:
                domains = {index[0]: domains}
            else:
                raise TypeError("provide domains as a dictionary of {i: domains_list} where i is in ic-numbering")
        elif type(domains) != dict:
            raise TypeError("provide domains as a dictionary of {i: domains_list} where i is in ic-numbering. Or, if len(i) == 1, provide domains_list")
        var = vds[var] if var in vds.keys() else var
        if "dih" in var:
            indexes = [("dih_c", [i for i in index if i in self.use_c]), ("dih", [i for i in index if i not in self.use_c])]
        else:
            indexes = [(var, index)]
        basins, centers = {}, {}
        for z in indexes:
            v, idx = z
            orig = getattr(self, vdp[v]) if v in ["bond", "angle"] else getattr(self, v)
            for i in idx:
                dmncntrs = np.array([orig[i][domain].mean() for domain in domains[i]])
                o = np.argsort(dmncntrs)
                d = np.append(np.nan, np.diff(dmncntrs[o]))
                mask = np.where(d < thresh)[0]
                cntr = 0
                within_l = []
                while cntr < mask.shape[0]:
                    start = cp.copy(mask[cntr]) - 1
                    end = cp.copy(start) + 2
                    while cntr + 1 < mask.shape[0] and mask[cntr+1] == mask[cntr] + 1:
                        cntr += 1
                        end += 1
                    within_l.append(o[start:end])
                    cntr += 1
                if within_l:
                    center_range = [dmncntrs[j].max()-dmncntrs[j].min() for j in within_l]
                    if max(center_range) > check_thresh:
                        raise BaseException("i = {}, range of domain centers in basin larger than check_thresh. cannot determine basins in this case".format(i))
                flat = list(it.chain.from_iterable(within_l))
                others = [j for j in o if j not in flat]
                basins[i] = [np.concatenate([domains[i][j] for j in k]) for k in within_l] + [domains[i][j] for j in others]
                centers[i] = [orig[i][b].mean() for b in basins[i]]
        return basins, centers
            
    def __getitem__(self, key):
        """
        Parameters
        ----------
        key: slice/int
            the slicing index(es)
        
        Returns
        -------
        if key is int, cc.zmat
        if key is slice, ic_avg
        """
        if type(key) in [int, np.int32, np.int64]:
            sliced = self.zmat1.copy()
            sliced._frame["bond"] = self.bonds[:,key] if self.avg_bond_angles == False else self.bonds
            sliced._frame["angle"] = self.angles[:,key] if self.avg_bond_angles == False else self.angles
            sliced._frame["dihedral"] = self.dih[:,key]
        else:
            slice_ = slicestr(key)
            source = "from {} with {}".format(id(self), slice_)
            sliced = Ic_averager(source, self.bonds[:,key], self.angles[:,key], self.dih[:,key], self.zmat1, self.c_table)
            sliced.zmat1._frame["bond"] = sliced.bonds[:,0] if self.avg_bond_angles == False else sliced.bonds  # updating zmat1 to be as frame 0
            sliced.zmat1._frame["angle"] = sliced.angles[:,0] if self.avg_bond_angles == False else sliced.angles  # updating zmat1 to be as frame 0
            sliced.zmat1._frame["dihedral"] = sliced.dih[:,0]  # updating zmat1 to be as frame 0
            if hasattr(self,"_dih_c"):
                setattr(sliced, "_dih_c", getattr(self, "_dih_c")[:,key])
            builtin = ["bonds", "angles", "dih", "source", "avg_bond_angles",
                       "c_table", "zmat1", "zmat", "cart", "natoms", "nframes",
                       '_dih_c', '_dih_c_std', '_dih_std', '_use_c',
                       "_dih_mean", "_dih_c_mean", "use_c"]
            pseudos = [i for i in self.__dict__.keys() if i not in builtin]
            pseudos = [i for i in pseudos if type(getattr(self,i)) == np.ndarray]  # only if arrays
            for pseu in pseudos:
                print(pseu)
                orig = getattr(self,pseu)
                new = orig[list(key)] if type(key)== tuple else orig[key] if len(orig.shape) == 1 else orig[:,key]
                setattr(sliced, pseu, new)
        return sliced
    
    def __repr__(self):
        """
        Note
        ----
        Representation for ic_avg, returns natoms, nframes, zmat1
        """
        string = "{} atoms for {} frames".format(self.natoms,self.nframes)
        frame = self.zmat1._frame.__str__()
        return "{}\n{}".format(string, frame)
        
    
    def copy(self):
        """Copies the Ic_averager        
        """
        return cp.deepcopy(self)
    
    def cart_to_ic(self, arr: Union[int, list, tuple, np.array]):  # test combination with plotting
        """
        Note
        ----
        Returns the array but in ic-numbering
        
        Parameters
        ----------
        arr: int/np.array/list/tuple
            the index(es) to convert
        
        Returns
        -------
        np.array
            the array in ic-numbering
        """
        if type(arr) in [int, np.int32, np.int64]:
            arr = [arr]
        return np.array([self.c_table.index.get_loc(i) for i in arr])
    
    def ic_to_cart(self, arr: Union[int, list, tuple, np.array]):
        """
        Note
        ----
        Returns the array but in cartesian-numbering
        
        Parameters
        ----------
        arr: int/np.array/list/tuple
            the index(es) to convert
        
        Returns
        -------
        np.array
            the array in cartesian-numbering
        """
        if type(arr) in [int, np.int32, np.int64]:
            arr = [arr]
        return np.array(self.c_table.index[arr])
    
    @classmethod
    def from_arrays(cls, atoms: Union[np.ndarray, list], arr: np.ndarray, 
                    source: str = "", int_coord_file: str = "internal_coordinates.npz",
                    save: bool =True, avg_bond_angles: bool = False, dec_digits: int = 3):  
        """Retrieves all internal coordinates from aligned trajectory in cartesians.
    
        Parameters
        ----------
        aligned_fn : str
            Filename of the fragment aligned over all trajectory.
        int_coord_file : str
            Filename for writing all the internal coordinates. ".npy" and ".npz" can be
            detected, otherwise ".txt" separate files are used.
        save : bool
            whether coordinates should be also saved (True) or just returned (False)
        avg_bond_angles: bool
            whether bonds and angles should already be averaged (for saving and returning)
        dec_digits : int
            Used only for ".txt" files, number of decimal digits
        """
        if not source:
            source = "from cartesian arrays, saved in {}".format(int_coord_file)
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
            if int_coord_file[-4:] == ".npy":
                np.save(int_coord_file, np.array([bonds, angles, dih]))
                print("saved bonds, angles, dihedrals in {}".format(int_coord_file))
            elif int_coord_file[-4:] == ".npz":
                np.savez(int_coord_file, bonds=bonds, angles=angles, dihedrals=dih)
                print("saved bonds, angles, dihedrals in {}".format(int_coord_file))
            else:
                int_coord_file += ".txt" if "txt" not in int_coord_file else ""
                np.savetxt(int_coord_file[:-4] + "_bonds.txt", np.array(bonds), fmt=fmt)
                np.savetxt(int_coord_file[:-4] + "_angles.txt", np.array(angles), fmt=fmt)
                np.savetxt(int_coord_file[:-4] + "_dihedrals.txt", np.array(dih), fmt=fmt)
                print("saved bonds,angles,dihedrals in {} respectively".format(", ".join(
                    [int_coord_file[:-4] + i + int_coord_file[-4:] for i in["_bonds", "_angles", "_dihedrals"]])))
        return cls(source, bonds, angles, dih, zmat1, c_table)
        
    @classmethod
    def from_aligned_cartesian_file(cls, aligned_fn: str = "aligned0.xyz", 
                                    int_coord_file: str = "internal_coordinates.npz",
                                    source: str = "", save: bool =True,
                                    avg_bond_angles: bool = False,
                                    c_table: Union[pd.DataFrame, bool] = False,
                                    dec_digits: int = 3):
        """Retrieves all internal coordinates from aligned trajectory in cartesians.
    
        Parameters
        ----------
        aligned_fn : str
            Filename of the fragment aligned over all trajectory.
        int_coord_file : str
            Filename for writing all the internal coordinates. ".npy" and ".npz" can be
            detected, otherwise ".txt" separate files are used.
        save : bool
            whether coordinates should be also saved (True) or just returned (False)
        avg_bond_angles: bool
            whether bonds and angles should already be averaged (for saving and returning)
        c_table: pd.DataFrame
            c_table to use if a specific one is needed
        dec_digits : int
            Used only for ".txt" files, number of decimal digits
        """
        t1 = time.time()
        if not source:
            source = "from {}. saved in {}".format(aligned_fn, int_coord_file)
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
                if c_table is False:
                    cart = str2cart(frame_str)
                    c_table = cart.get_construction_table()
                if f == 0:
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
            if int_coord_file[-4:] == ".npy":
                np.save(int_coord_file, np.array([bonds, angles, dih]))
                print("saved bonds, angles, dihedrals in {}".format(int_coord_file))
            elif int_coord_file[-4:] == ".npz":
                np.savez(int_coord_file, bonds=bonds, angles=angles, dihedrals=dih)
                print("saved bonds, angles, dihedrals in {}".format(int_coord_file))
            else:
                int_coord_file += ".txt" if "txt" not in int_coord_file else ""
                np.savetxt(int_coord_file[:-4] + "_bonds.txt", np.array(bonds))
                np.savetxt(int_coord_file[:-4] + "_angles.txt", np.array(angles))
                np.savetxt(int_coord_file[:-4] + "_dihedrals.txt", np.array(dih))
                print("saved bonds,angles,dihedrals in {} respectively".format(", ".join(
                    [int_coord_file[:-4] + i + int_coord_file[-4:] for i in["_bonds", "_angles", "_dihedrals"]])))
            source += "Internal coordinates saved in {}".format(int_coord_file)
        return cls(source, bonds, angles, dih, zmat1, c_table)
    
    @classmethod
    def from_int_coord_file(cls, source: str = "", aligned_fn: str = "",
                            int_coord_file: str = "", c_table: Union[pd.DataFrame, bool] = False):
        """ Retrieves all internal coordinates from file
        Parameters
        ----------
        aligned_fn : str
            Filename of the fragment aligned over all trajectory. Needed for zmat1, c_table
        int_coord_file : str
            File with all the internal coordinates. ".npy" and ".npz" can be
            detected, otherwise ".txt" separate files are used.
        c_table: pd.DataFrame
            the c_table to use if a specific one is needed
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
        c_table = cart.get_construction_table() if c_table is False else c_table
        zmat1 = str2zmat(frame_str, c_table)
        if not source:
            source = "from {}.".format(int_coord_file)
        return cls(source, bonds, angles, dih, zmat1, c_table)
    
    def plot_time_evo(self, index: Union[list,int], basins: Union[bool, np.ndarray] = False,
                      var: str = "dihedral", title: str = "", pos_range: Union[bool, type(None)] = None,
                      label: Union[list, str] = [], legend: bool = True,
                      displacement: bool = False, ravg: bool = False,
                      res_displ: int = 15, res_ravg: int = 10, recalculate: bool = False):
        """
        Note
        ----
        Plots the time evolution of one or more variable
        
        Parameters
        ----------
        index: list or int
            index or list of indexes to plot (data[index,:])
        basins: bool
            whether the basins should be plotted with different colors
        var: str 
            variable type (dih/angle/bond). also used as label for y axis. default is "dihedral"
        title: str
            title for the plot. default is empty string
        pos_range: bool
            whether to plot in the (-180,180) or in the (0,360) range.
            Only applies for var=="dih". 
        label: list/str
            label(s) for the series to plot. length must match that of index
        legend: bool
            whether the legend should be added
        displacement: bool
            whether the displacement should be plotted
        ravg: bool
            whether the running average should be plotted
        res_displ: int
            the size of the chunks for the displacement
        res_ravg: int
            the time interval for the running average
        recalculate: bool
            whether running average and/or displacement should be recalculated
            (use if changing res)
    
        Returns
        -------
        plt.figure
            the value vs t of data[i,:] for i in index
        """
        var = vdp[var] if var in vdp.keys() else var
        index = index if type(index) in [list, np.ndarray] else [index]  # make list if is not
        if var == "dih" and pos_range is None:
            pos_range = True if sum([1 if i in self.use_c else -1 for i in index])> 0 else False  # looks at all dihedrals
            data = getattr(self, "dih_c" if pos_range else "dih")
        else:
            data = getattr(self, var)
        if not label:
            label = list(self.c_table.index[index])
        if type(label) != list:
            label = [label]
        fig = plot_time_evo(data, index, basins=basins, var=var, title=title, pos_range=pos_range, label=label, legend=legend)
        if displacement or ravg:
            marker = "P" if displacement and ravg else "o"
            markersize = 5 if displacement and ravg else 1
            ax = fig.axes[0]
            if "dih" in var:
                indexes = [
                        ["dih_c", *list(zip(*[(i, label[n]) for n,i in enumerate(index) if i in self.use_c]))],
                         ["dih", *list(zip(*[(i, label[n]) for n,i in enumerate(index) if i not in self.use_c]))]
                         ]
            else:
                indexes = [[var, index, label]]
            for j in indexes:
                if len(j) != 3:
                    continue
                v, idx, labels = j
                ### DISPLACEMENT
                if displacement:
                    if recalculate or not hasattr(self, "{}_d".format(vds[v])):
                        self.get_displacement(var=v, index=idx, res=res_displ)
                        d = getattr(self, "{}_d".format(vds[v]))
                    else:
                        d = getattr(self, "{}_d".format(vds[v]))
                        missing = [i for i in idx if i not in d.keys()]
                        if missing:
                            self.get_displacement(var=v, index=missing, res=res_displ)
                    for n,i in enumerate(idx):
                        ax.plot(d[i][1]*(1.5+np.arange(d[i][0].shape[0])), d[i][0], label="{} displ".format(labels[n]),
                        marker=marker, markersize=markersize, markeredgecolor="k", linewidth=0.5, linestyle="--")  # d[i] = (disp_arr, res)
                print("marker")
                ### RAVG
                if ravg:
                    if recalculate or not hasattr(self, "{}_m".format(vds[v])):
                        self.get_ravg(var=v, index=idx, res=res_ravg)
                        d = getattr(self, "{}_m".format(vds[v]))
                    else:
                        d = getattr(self, "{}_m".format(vds[v]))
                        missing = [i for i in idx if i not in d.keys()]
                        if missing:
                            self.get_ravg(var=v, index=missing, res=res_ravg)
                    for n,i in enumerate(idx):
                        ravg_arr = d[i][0]
                        ax.plot(np.arange(ravg_arr.shape[0]), ravg_arr, label="{} ravg".format(labels[n]),
                        marker="o", markersize=1, markeredgecolor="k", linewidth=0.5, linestyle="--")  # d[i] = (disp_arr, res)
            if var in ["dih", "dih_c", "angle"]:
                ax.set_ylim(-180, ax.get_ylim()[1])
            if legend:
                ax.legend()
        return fig
    
    def plot_distrib(self, index: Union[list,int], basins: Union[bool, np.ndarray] = False,
                     bins: int = 36, var: str = "dihedral", title: str = "",
                     pos_range: bool = None, label: Union[list,str] = [], alpha: float = 0.0,
                     legend: bool = True):
        """
        Note
        ----
        Plots the distribution of one or more variable
        
        Parameters
        ----------
        index: list or int
            index or list of indexes to plot (data[index,:])
        bins: int
            number of bins for histogram. default is 36
        var: str 
            variable type (dih/angle/bond). also used as label for y axis. default is "dihedral"
        title: str
            title for the plot. default is empty string
        pos_range: bool
            whether to plot in the (-180,180) or in the (0,360) range.
            Only applies for var=="dih". 
        label: list/str
            label(s) for the series to plot. length must match that of index
        alpha: float
            transparency. default uses 1.0 for 1 distribution, 0.75 for 2, 0.5 for more
            
        Returns
        -------
        plt.figure
            the histogram with distribution of data[i,:] for i in index
        """
        index = index if type(index) in [list, np.ndarray] else [index]  # make list if is not
        var = vdp[var] if var in vdp.keys() else var
        if var == "dih" and pos_range is None:
            pos_range = True if sum([1 if i in self.use_c else -1 for i in index])> 0 else False  # looks at all dihedrals
            data = getattr(self, "dih_c" if pos_range else "dih")
        else:
            data = getattr(self, var)
        if not label:
            label = list(self.c_table.index[index])
        return plot_distrib(data, index, basins=basins, bins=bins, var=var, title=title,
                            pos_range=pos_range, label=label, alpha=alpha, legend=legend)
        
    def plot_2Ddistrib(self, index: list, var: Union[list, tuple, str] = "dihedral", bins: int = 36,
                 labels: list= [],  title: str = "",
                 pos_range: Union[list,bool] = None, label: Union[list,str] = []):
        """Plots value occurrence vs value
        Parameters
        ----------
        index: list/int
            index or list of indexes to plot (data[index,:])
        var: list/tuple/str
            variable type(s). used for range and axis labels
        bins: int
            number of bins for histogram. default is 36
        y_label: str 
            label for y axis. default is "dihedral"
        title: str
            title for the plot. default is empty string
        pos_range: list[bool]/bool
            whether to plot in the (-180,180) or in the (0,360) range on each axis.
            If boolean, turns to list of twice that value
        
        Returns
        -------
        plt.figure
            the histogram with distribution of data[i,:] for i in index
        """
        if type(var) == str:
            var = [var, var]
        var = [vdp[v] if v in vdp.keys() else v for v in var]
        if pos_range is None:
            pos_range = [None, None]
            data = [None, None]
            for n,v in enumerate(var):
                if v == "dih":
                    pos_range[n] = index[n] in self.use_c
                    data[n] = self.dih_c if pos_range[n] else self.dih
                else:
                    data[n] = getattr(self, v)
        if not label:
            label = list(self.c_table.index[index])
        return plot_2Ddistrib(data, index, var=var, bins=bins, title=title, pos_range=pos_range, label=label)
    
    def plot_displacement(self, index: Union[list,int], var: str = "dihedral",
                          title: str = "displacement", label: Union[list, str] = [],
                          legend: bool = True, res: int = 15, recalculate: bool = False):
        """
        Parameters
        ----------
        index: list/int
            indexes to plot
        var: str
            the variable (bond, angle, dih, dih_c, some pseudo)
        title: str
            optional fig title
        label: list/str
            optional label for the different indexes plotted ()
        legend: bool
            whether the legend should appear or not
        res: int
            resolution if some displacement needs to be calculated
        recalculate: bool
            whether running average and/or displacement should be recalculated
            (use if changing res)
        """
        var = vds[var] if var in vds.keys() else var
        if type(index) in [int, np.int32, np.int64]:
            index = [index]
        if not label:
            label = [str(i) for i in list(self.c_table.index[index])]
        elif type(label) in [str,int, np.int32, np.int64]:
            label = [label]
        if "dih" in var:
            indexes = [
                    ["dih_c", *list(zip(*[(i, label[n]) for n,i in enumerate(index) if i in self.use_c]))],
                     ["dih", *list(zip(*[(i, label[n]) for n,i in enumerate(index) if i not in self.use_c]))]
                     ]
        else:
            indexes = [[var, index, label]]
        fig = plt.figure(figsize=(20, 10), dpi=150)
        ax = fig.add_subplot(111)
        for j in indexes:
            if len(j) != 3:
                continue
            v, idx, labels = j
            if recalculate or not hasattr(self, "{}_d".format(vds[v])):
                self.get_displacement(var=v, index=idx, res=res)
                d = getattr(self, "{}_d".format(vds[v]))
            else:
                d = getattr(self, "{}_d".format(vds[v]))
                missing = [i for i in idx if i not in d.keys()]
                if missing:
                    self.get_displacement(var=v, index=missing, res=res)
            
            for n,i in enumerate(idx):
                ax.plot(d[i][1]*(1.5+np.arange(d[i][0].shape[0])), d[i][0], label=labels[n],  # d[i] = (disp_arr, res)
                        marker="o", markersize=1, markeredgecolor="k", linewidth=0.5, linestyle="--")
        ax.set_ylabel(var.replace("_","\_"))  
        ax.set_xlabel("frame")
        if legend:
            ax.legend()
        ax.set_title(title)
        return fig

    def plot_ravg(self, index: Union[list,int], var: str = "dihedral",
                          title: str = "displacement", label: Union[list, str] = [],
                          legend: bool = True, res: int = 15, recalculate: bool = False):
        """
        Parameters
        ----------
        index: list/int
            indexes to plot
        var: str
            the variable (bond, angle, dih, dih_c, some pseudo)
        title: str
            optional fig title
        label: list/str
            optional label for the different indexes plotted ()
        legend: bool
            whether the legend should appear or not
        res: int
            resolution if some displacement needs to be calculated
        recalculate: bool
            whether running average and/or displacement should be recalculated
            (use if changing res)
        """
        var = vds[var] if var in vds.keys() else var
        if type(index) in [int, np.int32, np.int64]:
            index = [index]
        if not label:
            label = [str(i) for i in list(self.c_table.index[index])]
        elif type(label) in [str,int, np.int32, np.int64]:
            label = [label]
        if "dih" in var:
            indexes = [
                    ["dih_c", *list(zip(*[(i, label[n]) for n,i in enumerate(index) if i in self.use_c]))],
                     ["dih", *list(zip(*[(i, label[n]) for n,i in enumerate(index) if i not in self.use_c]))]
                     ]
        else:
            indexes = [[var, index, label]]
        fig = plt.figure(figsize=(20, 10), dpi=150)
        ax = fig.add_subplot(111)
        for j in indexes:
            if len(j) != 3:
                continue
            v, idx, labels = j
            if recalculate or not hasattr(self, "{}_m".format(vds[v])):
                self.get_ravg(var=v, index=idx, res=res)
                d = getattr(self, "{}_m".format(vds[v]))
            else:
                d = getattr(self, "{}_m".format(vds[v]))
                missing = [i for i in idx if i not in d.keys()]
                if missing:
                    self.get_ravg(var=v, index=missing, res=res)
            for n,i in enumerate(idx):
                ax.plot(d[i][1]*(1.5+np.arange(d[i][0].shape[0])), d[i][0], label=labels[n],  # d[i] = (disp_arr, res)
                        marker="o", markersize=1, markeredgecolor="k", linewidth=0.5, linestyle="--")
        ax.set_ylabel(var.replace("_","\_"))  
        ax.set_xlabel("frame")
        if legend:
            ax.legend()
        ax.set_title(title)
        return fig
    
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
            if not hasattr(self,"_use_c"):
                diff_std = self.dih_c_std - self.dih_std
                self._use_c = np.where(diff_std < 0)[0]  # where it's better to use 0-360 range
                
    def std_analysis(self, var="dih", thresh_min: Union[float, int] = 90, thresh_max: Union[float, int] = 180, group_name: Union[str,bool] = ""):
        """
        Note
        ----
        Generally 90-180 works for freely rotating groups, while 37/40-90 works for conformational equilibrium (tested on cyclohexene in retinal)
        Parameters
        ----------
        var: str
            variable type ("dih"/"dihedral", angle, ...)
        thresh_min: float or int
            threshold for minimum std for the dihedral on the most appropriate range
        thresh_max: float or int
            threshold for maximum std for the dihedral on the most appropriate range
        group_name str
            name to use for the groups found
                
        Sets
        ----
        self.[group_name]: list[Groups]
            list of detected Groups
        
        Returns
        -------
        list
            group_list if group_name is an empty string or False
        """
        var = vdp[var] if var in vdp.keys() else var 
        if var == "dih":
            detected = np.arange(3,self.natoms)[np.logical_and(np.minimum(self.dih_std[3:], self.dih_c_std[3:]) > thresh_min,np.minimum(self.dih_std[3:], self.dih_c_std[3:]) < thresh_max)]
            group_list, done = [], []
            for i  in detected:
                if i not in done:
                    b2s_cart = self.c_table[self.c_table["b"]==self.c_table.iloc[i]["b"]].index  # bound to the same, cartesian numbering
                    b2s_arr = [self.c_table.index.get_loc(j) for j in b2s_cart]  # passing to arr numbering
                    done.extend(b2s_arr)
                    gr = Group(b2s_arr, id(self), var="dih")
                    setattr(gr,"_cart", b2s_cart)  # not "gr.cart = because @property
                    group_list.append(gr)
        elif var in ["bond", "angle"]:
            shift = 2 if var == "angle" else 1
            group_list = np.arange(shift,self.natoms)[np.logical_and(getattr(self,var+"s").std(axis=1) > thresh_min, getattr(self,var+"s").std(axis=1) < thresh_max)]  # "bond"=> "bonds", "angle"=> "angles"
        else: # pseudo
            group_list = np.arange(shift,self.natoms)[np.logical_and(getattr(self,var).std(axis=1) > thresh_min, getattr(self,var).std(axis=1) < thresh_max)]
        if group_name:    
            setattr(self, group_name, group_list)
        else:
            return group_list
        print("Obtained {} with thresholds: [{},{}]".format(group_name, thresh_min, thresh_max))
        print("Resulting in groups:\n{}".format("\n".join([str(i) for i in group_list])))
    
    def find_clusters(self, var: str = "", index: Union[None, int] = None, min_centers: int = 1, max_centers: int =3):
        """
        Note
        ----
        Clustering ML algorithm. Slight randomness due to starting values.
        
        Parameters
        ----------
        var: str
            variable type ("dih", "angle", "bond")
        index: int
            atom to analyse
        min_centers: int
            minimum number of centers
        max_centers: maximum number of centers
        
        Returns
        -------
        tuple
            basins (list of lists), centers(list)
        
        """
        # TODO make indexes a list
        using_c = False
        if index is None:
            raise ValueError("You must specify \"index\".")
        if var == "":
            print("no variable type specified, supposing it is \"dihedral\"")
            var = "dih"
        var = vdp[var] if var in vdp.keys() else var
        if var == "dih":
            if hasattr(self, "use_c"):
                (arr, using_c) = (self.dih_c[index],True) if index in self.use_c else (self.dih[index], False)
            else:
                print("Watch out! No correction of quasilinear dihedrals has been performed thus far!")
                arr = self.dih[index]
        elif var in ["angle", "bond"]:
            arr = getattr(self, var+"s")
        else: 
            arr = getattr(self, var)
        arr = arr.reshape(-1,1)
        initial_centers = kmeans_plusplus_initializer(arr, min_centers).initialize()
        xmeans_instance = xmeans(arr, initial_centers, max_centers)
        xmeans_instance.process()
        basins = list(map(np.asarray, xmeans_instance.get_clusters()))
        centers = [conv_d(i) for i in list(it.chain.from_iterable(xmeans_instance.get_centers()))] if using_c else list(it.chain.from_iterable(xmeans_instance.get_centers()))
        return basins, centers
    
    def get_bins(self, var: str, index: int, bins: int = 5, range_: tuple = ()):  # test
        """
        Note
        ----
        creates bins (as many as asked) of equal value-range within range(if not provided is range of your values).
        Then the indexes of the frames in each basin.
        
        Parameters
        ----------
        var: str
            type of variable (dih/angle/bond)
        index: int
            index (ic-numbering) of the variable to base bins on
        bins: int
            number of bins desired
        range_: tuple
            (min,max). Default looks at min/max of your values.
            Otherwise, values outside the range are ignored
        """
        from scipy.sparse import csr_matrix
        var = vdp[var] if var in vdp.keys() else var
        values = getattr(self, var)[index,:]
        frames = np.arange(self.nframes)
        shape = [bins, len(frames)]  # shape can be inferred but it is probably faster to give it
        includes_max = False
        if not range_:
            range_ = (values.min(), values.max())
        else:
            in_range = np.logical_and(values >= range_[0], values <= range_[1])
            frames = frames[in_range]  # frame numbers
            values = values[in_range]
        digitized = (float(bins)/(range_[1] - range_[0])*(values - range_[0])).astype(int)  # array of what bin each frame is in
        if bins in digitized:
            digitized[digitized == bins] = bins -1  # so that last bin includes max
            shape[0] += 1
            includes_max = True
        S = csr_matrix((values, [digitized, frames]), shape=shape)
        to_return = np.split(S.indices, S.indptr[1:-1])[:-1] if includes_max else np.split(S.indices, S.indptr[1:-1])
        # last bin empty (only max values but we moved them to second to last bin)
        return to_return
    
    def select_remaining(self, groupset_name: str, criterion: Union[str, callable]):
        """
        Parameters
        ----------
        groupset_name: str
            group to act on (rotate, oscillate, ...)
        criterion: str/callable
        
        Sets
        ----
        group._selected for any group in self.groupset_name which did not have such attribute
        """
        groupset = getattr(self, groupset_name)
        for gr in groupset:
            if not hasattr(gr, "_selected"):
                gr.select(criterion)
            
    def average(self, out_file: str = "averaged.xyz", overwrite: bool = False,
                           view: bool = True, viewer: str = "avogadro",
                           basin=1):  
        """Averages the internal coordinates. Rotate/ing refers to fully/freely rotating groups,
        oscillate/ing refers to groups oscillating between two conformers.
        
        Parameters
        ----------
        out_file: str
            where to write the averaged structure. default is "averaged.xyz"
        overwrite: bool
            whether out_file can be overwritten. if False it creates averaged_n.xyz with n=1,2...
            
        Sets
        -------
        self.zmat
            cc.Zmat of the averaged structure
        """
        self.zmat = self.zmat1.copy()
        self.zmat._frame["bond"] = self.bonds if self.avg_bond_angles else self.bonds.mean(axis=1) 
        self.zmat._frame["angle"] = self.angles if self.avg_bond_angles else self.angles.mean(axis=1)
        self.zmat._frame["dihedral"] = self.dih_mean
        if hasattr(self, "_use_c"):
            self.zmat._frame["dihedral"].values[self._use_c] = self.dih_c_mean[self._use_c]
        builtin = ["bonds", "angles", "dih", "source", "avg_bond_angles", "c_table", "zmat1", "zmat", "cart", "natoms", "nframes"]
        try:
            groupsets = [i for i in self.__dict__.keys() if i not in builtin]  # no builtins
            groupsets = [i for i in groupsets if type(getattr(self,i))==list and type(getattr(self,i)[0]) == Group]  # only if first element is a group
        except BaseException:
            groupsets = []
            print("no groupsets")
        for gs in groupsets:
            var = vdf[getattr(self,gs)[0].var] if getattr(self,gs)[0].var in vdf.keys() else False  # False if not dih,bond,angle
            if not var:
                continue  # we do not average pseudos
            else:
                other_vars = [i for i in ["bond", "angle", "dihedral"] if i != var]
            for gr in getattr(self,gs):
                avg_vals, basin_frames = gr.get_avg_values(basin=basin, return_frames=True)  # avg_vals from group
                self.zmat._frame[var].values[gr.arr] = avg_vals  # avg_vals for the group variable
                for ov in other_vars:
                    self.zmat._frame[ov].values[gr.arr] = getattr(self, vdp[ov])[gr.arr,:][:,basin_frames].mean(axis=1)  # other vars averaged over the basin only
        self.cart = self.zmat.get_cartesian()
        if out_file:
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
    @property 
    def use_c(self):
        if not hasattr(self,"_use_c"):
            self.correct_quasilinears()
        return self._use_c


