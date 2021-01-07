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
from ctypes import cast, py_object
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


rc('text', usetex=True)
vdict = {"dihedral": "dih", "d": "dih", "dih": "dih",
         "angle": "angle", "angles": "angle", "a": "angle", 
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
    
def slicestr(slc: slice):
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
    start = slc.start if slc.start else ""
    stop = slc.stop if slc.stop else ""
    step = slc.step if slc.step else ""
    return "{}:{}:{}".format(start, stop, step)

def plot_2Ddistrib(data: np.ndarray, index: list, var: Union[list, str, tuple] = "dihedral",
                   bins: int = 36, labels: list= [],  title: str = "",
                 pos_range: Union[list,bool] = True, label: Union[list,str] = []):
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
    pos_range: list[bool]/bool
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
    var = [vdict[v] for v in var]
    if type(pos_range) == bool:
        pos_range = [pos_range, pos_range]
    range_ = [None, None]  # check if works for bonds
    for n,v in enumerate(var):
        if v == "dih":
         range_[n] = (0, 360) if pos_range[n] else (-180, 180)
        elif var == "angle":
            var[n] = (0,180)
    ax.hist2d(data[index[0], :], data[index[1], :], bins=[bins,bins], range=range_)
    if not label:
        ax.set_xlabel("{} {}".format(var[0],index[0]))
        ax.set_ylabel("{} {}".format(var[1],index[1]))
    else:
        ax.set_xlabel("{}".format(label[0]))
        ax.set_ylabel("{}".format(label[1]))
    ax.set_title(title)
    return fig

def plot_distrib(data: np.ndarray, index: Union[list,int], bins: int = 36,
                 var: str = "dihedral", title: str = "",
                 pos_range: bool = True, label: Union[list,str] = [],
                 alpha: float = 0.0):
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
        
    Returns
    -------
    plt.figure
        the histogram with distribution of data[i,:] for i in index
    """
    fig = plt.figure(figsize=(20, 10), dpi=150)
    var = vdict[var] if var in vdict.keys() else var
    ax = fig.add_subplot(111)
    if type(index) == int:
        index = [index]
    if not label:
        label = [str(i) for i in index]
    elif type(label) == str:
        label = [label]
    if var == "dih":
        range_ = (0, 360) if pos_range else (-180, 180)
    elif var == "angle":
        range_ = (0,180)
    if not alpha:
        alpha = 1.0 if len(index) == 1 else 0.75 if len(index) == 2 else 0.5
    for n,i in enumerate(index):
        ax.hist(data[i, :], bins=bins, range=range_, label=label[n])
    ax.set_ylabel("occurrence") 
    ax.set_xlabel(var)
    ax.set_title(title)
    return fig


def plot_time_evo(data: np.ndarray, index: Union[list,int], var: str = "dihedral",  
                  title: str = "", pos_range: bool = True, label: Union[list,str] = []):
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
    
    Returns
    -------
    plt.figure
        the value vs t of data[i,:] for i in index
    """
    x = np.arange(data.shape[1])
    var = vdict[var] if var in vdict.keys() else var
    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    if type(index) == int:
        index = [index]
    if not label:
        label = [str(i) for i in index]
    elif type(label) == str:
        label = [label]
    for n,i in enumerate(index):
        ax.scatter(x, data[i, :], label=label[n])
    ax.set_ylabel(var)  
    ax.set_xlabel("frame")
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

class group:
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
            uses vdict for dihedral/dih, angles/angle, etc...
        """
        if type(arr) in [list,tuple]:
            arr = np.array(arr)
        self.arr = arr
        self.avg_id = avg_id  # id(ic_avg), works as pointer to the averager
        self.var = vdict[var] if var in vdict.keys() else var
        
    def __repr__(self):
        """
        Note
        ----
        When calling group, it returns a string with ic-numbering and, if available, cartesian counting
        """
        if hasattr(self, "cart"):
            return "IC: {}, Cart: {}".format(self.arr.tolist(), self.cart.tolist())
        else:
            return "IC: {}".format(self.arr.tolist())
        
    def __str__(self):
        """
        Note
        ----
        When converting group to a string (e.g. printing), it returns a string with ic-numbering and, if available, cartesian counting
        """
        if hasattr(self, "cart"):
            return "IC: {}, Cart: {}".format(self.arr.tolist(), self.cart.tolist())
        else:
            return "IC: {}".format(self.arr.tolist())
        
    @staticmethod
    def from_cart(cart: np.array, avg_id: int):
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
        gr = group(arr, avg_id)
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
            avg = cast(self.avg_id, py_object).value
            basins, centers = [], []
            for i in self.arr:
                b, c = avg.find_clusters(var=self.var, index=i, min_centers=min_centers, max_centers=max_centers)
                basins.append(b), centers.append(c)
            self._basins, self._centers = basins, centers
            
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
        if hasattr(self, "_selected") and self._selected != idx:
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
            
    def select_prominence(self, minormax: Union[str, callable]):  # Warning: possible issue with "callable"
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
            self.select_least_prominent
        elif minormax in ["max", "maximum", "most", "more", max]:
            self.select_most_prominent
        else:
            raise ValueError("Use min/minimum/least/less or max/maximum/most/more")
            
    def res_analysis(self):
        """
        Note
        Analysis of "residuals". 
        i.e. looks for the combination of basins which minimises difference between 
        angle differences for the basins centers and average angle differences over the trajectory
        """
        if len(self.centers) == 1:
            res = [0 for c in self.centers[0]]
        else:
            avg = cast(self.avg_id, py_object).value
            res = [[0 for i in j] for j in self.centers]
            for na,angle in enumerate(self.centers):
                others = [(m,i) for m,i in enumerate(self.arr) if m != na]  # (numbering in group, numbering in ic)
                mdiffs = [(conv_d(avg.dih[self.arr[na],:] - avg.dih[self.arr[i[1]],:])).mean() for i in others]
                for nc,c in enumerate(self.centers[na]):
                    for o in others:
                        expected = conv_d(c - mdiffs[o[0]])   # expected value based on avg diff
                        res[na][nc] += min([i  - expected for  i in self.centers[o[0]]])
            self._res = res
            
    @property
    def res(self):
        """
        Note
        ----
        Sum of residuals for each center
        """
        if not hasattr(self, "res"):
            self.res_analysis()
        return self._res
    
    def select_res(self):
        """
        Note
        ----
        Selects atom with the lowest residuals
        """
        self.selected = mindidx(self.res)[0]
    
    def time_evo_sep(self, title: str = ""):
        """
        Note
        ----
        Plots time evolution of each variable in group, each in a separate picture
        
        Parameters
        ----------
        title: str
            title for the plot. default is empty string
            
        Returns
        -------
        plt.figures
            the value vs t of data[i,:] for i in index
        """
        avg = cast(self.avg_id, py_object).value
        to_return = []
        for n,a in enumerate(self.arr):
            to_return.append(avg.plot_time_evo(a, var=self.var, title=title, label=self.cart[n]))
        return to_return
    
    def time_evo(self, title: str = ""):
        """
        Note
        ----
        Plots time evolution of each variable in group, all in the same picture
        
        Parameters
        ----------
        title: str
            title for the plot. default is empty string
        
        Returns
        -------
        plt.figure
            the value vs t of data[i,:] for i in index
        """
        avg = cast(self.avg_id, py_object).value
        return avg.plot_time_evo(self.arr.tolist, var=self.var, title=title, label=self.cart)
    
    def distrib_sep(self, bins: int = 36, title: str = "", alpha: float = 0.0):
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
            to_return.append(avg.plot_distrib(a, bins=bins, var=self.var, title=title, label=self.cart[n], alpha=alpha))
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
        return avg.plot_distrib(self.arr, bins=bins, var=self.var, title=title, label=self.cart)
            
    def get_avg_values(self, basin: int = 1):  # TODO: overwrite issues
        """
        Note
        ----
        Gets average values for the desired basin
        
        Parameters
        ----------
        basin: int/"res"
            if basin=n the n-th to last basin per weight is used (1=first basin, 2=second basin...)
            "res" uses the basin that minimises residual sums
            
        Returns
        -------
        the average values
        """
        if not hasattr(self,"_avg_values"):
            vals = []
            if type(basin) == int:
                tosub = self.centers[self.selected][nthtolast(self.weights[self.selected], basin)]  
            elif basin == "res":
                self.select_res
                tosub = self.centers[self.selected][mindidx(self.res)[1]]
            else:
                raise ValueError("unrecognised value for \"basin\": it must be an implemented method to select the basin")
            avg = cast(self.avg_id, py_object).value
            for n, basins in enumerate(self.basins):
                if n == self.selected:
                    vals.append(tosub)
                else:
                    diff = (conv_d(avg.dih[self.arr[self.selected],:] - avg.dih[self.arr[n],:])).mean()
                    vals.append(conv_d(tosub - diff))
            self._avg_values = vals
        return self._avg_values

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

def intersect_groups(grouplist: list, basin: int = 1):
    """
    Note
    ----
    Given a list of groups, returns the frames of the common basin,
    i.e. the intersection of the desired basins for the selected atoms of each group
    
    Parameters
    ----------
    basin: int/"res"
        if basin=n the n-th to last basin per weight is used (1=first basin, 2=second basin...)
        "res" uses the basin that minimises residual sums
    
    Returns
    -------
    list
        The intersection of these basins (list of framenumbers)
    """
    if type(basin) == int:
        get_frames = lambda gr: gr.basins[gr.selected][nthtolast(gr.weights[gr.selected], basin)]  
    elif basin == "res":
        def get_frames(gr):
            gr.select_res
            return gr.centers[gr.selected][mindidx(gr.res)[1]]
    ll = [get_frames(gr) for gr in grouplist]
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
    if type(basin) == int:
        get_frames = lambda gr: gr.basins[gr.selected][nthtolast(gr.weights[gr.selected], basin)]  
    elif basin == "res":
        def get_frames(gr):
            gr.select_res
            return gr.centers[gr.selected][mindidx(gr.res)[1]]
    ll = [get_frames(gr) for gr in grouplist]
    return agreement_lists(ll)

class ic_averager:
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
        self.source = source 
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
    
    def add_pseudo(self, name, arr):
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
            
        Sets
        ----
        self.name
        """
        nframes = arr.shape[0] if len(arr.shape) == 1 else arr.shape[1]
        if nframes != self.nframes:
            raise ValueError("The number of frames does not match!!")
        setattr(self, name, arr)
        
    def pseudos_from(self, var="", **kwargs):
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
        **kwargs: dict
            "pseudo_name"1: function1
            "pseudo_name2": function2
        
        Sets
        ----
        self.pseudo = np.array(k, nframes) 
            k depends on your function, most commonly 1
        """
        if var:  # we can use apply_along_axis
            var = vdict[var]
            arr = getattr(self, var) if var == "dih" else getattr(self+"s",var)
            for coord, func in kwargs.items():
                pseudo = np.apply_along_axis(func, 0, arr)
                if len(pseudo.shape) == 1:
                    pseudo = pseudo.reshape(1,-1)
                setattr(self, coord, pseudo)
        else:
            pseudo = np.zeros(self.nframes)
            for f in range(self.nframes):
                bonds, angles, dih = self.bonds[:, f], self.angles[:, f], self.dih[:,f]
                for coord, func in kwargs.items():
                    pseudo[f] = func(bonds=bonds, angles=angles, dih=dih)
            setattr(self, coord, pseudo)    
            
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
        if type(key) == int:
            sliced = self.zmat1.copy()
            sliced._frame["bonds"] == self.bonds[key] if self.avg_bond_angles == False else self.bonds
            sliced._frame["angles"] == self.angles[key] if self.avg_bond_angles == False else self.angles
            sliced._frame["dihedral"] = self.dih[key]
        else:
            slice_ = slicestr(key)
            source = "from {} with {}".format(id(self), slice_)
            sliced = ic_averager(source, self.bonds[key], self.angles[key], self.dih[key], self.zmat1, self.c_table)
        return sliced
    
    def __repr__(self):
        """
        Note
        ----
        Representation for ic_avg, returns natoms, nframes, zmat1
        """
        string = "{} atoms for {} frames".format(self.natoms,self.nframes)
        frame = self.zmat1.frame_.__str__()
        return "{}\n{}".format(string, frame)
        
    
    def copy(self):
        """Copies the ic_averager        
        """
        import copy as c
        return c.copy(self)
    
    def cart_to_ic(self, arr: Union[int, list, tuple, np.array]):
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
        return np.array(self.c_table.index[arr])
    
    @classmethod
    def from_arrays(cls, source, atoms: Union[np.ndarray,list], arr: np.ndarray, int_coord_file: str = "internal_coordinates.npz",
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
            source = "from arrays, saved in {}".format(int_coord_file)
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
#       
    @classmethod
    def from_aligned_cartesian_file(cls, aligned_fn: str = "aligned0.xyz", int_coord_file: str = "internal_coordinates.npz",
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
        return cls(aligned_fn, bonds, angles, dih, zmat1, c_table)
    
    @classmethod
    def from_int_coord_file(cls, aligned_fn: str = "", int_coord_file: str = ""):
        """ Retrieves all internal coordinates from file
        Parameters
        ----------
        aligned_fn : str
            Filename of the fragment aligned over all trajectory. Needed for zmat1, c_table
        int_coord_file : str
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
        return cls(int_coord_file, bonds, angles, dih, zmat1, c_table)
    
    def plot_time_evo(self, index: Union[list,int], var: str = "dihedral",
                      title: str = "", pos_range: bool = True,
                      label: Union[list, str] = []):
        """
        Note
        ----
        Plots the time evolution of one or more variable
        
        Parameters
        ----------
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
    
        Returns
        -------
        plt.figure
            the value vs t of data[i,:] for i in index
        """
        data = getattr(self,vdict[var])
        if vdict[var] == "dih":
            pos_range = self.use_c
        if not label:
            label = list(self.c_table.index[index])
        return plot_time_evo(data, index, var=var, title=title, pos_range=pos_range, label=label)
    
    def plot_distrib(self, index: Union[list,int], bins: int = 36,
                 var: str = "dihedral", title: str = "",
                 pos_range: bool = True, label: Union[list,str] = [], alpha: float = 0.0):
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
        data = getattr(self,vdict[var])
        if vdict[var] == "dih":
            pos_range = self.use_c
        if not label:
            label = list(self.c_table.index[index])
        return plot_distrib(data, index, bins=bins, var=var, title=title,pos_range=pos_range, label=label, alpha=alpha)
        
    def plot_2Ddistrib(self, index: list, var: Union[list, tuple, str] = "dihedral", bins: int = 36,
                 labels: list= [],  title: str = "",
                 pos_range: Union[list,bool] = False, label: Union[list,str] = []):
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
        var = [vdict[v] for v in var]
        data = getattr(self,vdict[var])
        if not pos_range:
            for n,v in enumerate(var):
                if v == "dih":
                    pos_range[n] = self.use_c[index[n]]
        if not label:
            label = list(self.c_table.index[index])
        return plot_2Ddistrib(data, index, bins=bins, title=title, pos_range=pos_range, label=label)

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
    
    def std_analysis(self, var="dih", thresh_min: Union[float, int] = 90, thresh_max: Union[float, int] = 180, group_name: str = "rotate"):
        """
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
        self.[group_name]: list[groups]
            list of detected groups
        """
        var = vdict[var] if var in vdict.keys() else var if var in vdict.keys() else var
        if var == "dih":
            detected = np.arange(3,self.natoms)[np.logical_and(np.minimum(self.dih_std[3:], self.dih_c_std[3:]) > thresh_min,np.minimum(self.dih_std[3:], self.dih_c_std[3:]) < thresh_max)]
            group_list, done = [], []
            for i  in detected:
                if i not in done:
                    b2s_cart = self.c_table[self.c_table["b"]==self.c_table.iloc[i]["b"]].index  # bound to the same, cartesian numbering
                    b2s_arr = [self.c_table.index.get_loc(j) for j in b2s_cart]  # passing to arr numbering
                    done.extend(b2s_arr)
                    gr = group(b2s_arr, id(self), var="dih")
                    setattr(gr,"cart", b2s_cart)  # not "gr.cart = because @property
                    group_list.append(gr)
        elif var in ["bond", "angle"]:
            shift = 2 if var == "angle" else 1
            group_list = np.arange(shift,self.natoms)[np.logical_and(getattr(self,var+"s").std(axis=1) > thresh_min, getattr(self,var+"s").std(axis=1) < thresh_max)]  # "bond"=> "bonds", "angle"=> "angles"
        else: # pseudo
            group_list = np.arange(shift,self.natoms)[np.logical_and(getattr(self,var).std(axis=1) > thresh_min, getattr(self,var).std(axis=1) < thresh_max)]
            
        setattr(self, group_name, group_list)
        print("Obtained {} with thresholds: [{},{}]".format(group_name, thresh_min, thresh_max))
        print("Resulting in groups:\n {}".format("    ".join(group_list)))
    
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
        using_c = False
        if index == None:
            raise ValueError("You must specify \"index\".")
        if var == "":
            print("no variable type specified, supposing it is \"dihedral\"")
            var = "dih"
        var = vdict[var] if var in vdict.keys() else var
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
    
    def get_bins(self, var: str, index: int, bins: int = 5, range_: tuple = ()):
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
        var = vdict[var] if var in vdict.keys() else var
        values = getattr(self, var+"s")[index,:] if var in ["bonds", "angles"] else getattr(self, var)[index,:]
        frames = np.arange(self.nframes)
        if not range_:
            range_ = (values.min(), values.max())
        else:
            in_range = np.logical_and(values >= range_[0], values <= range_[1])
            frames = frames[in_range]  # frame numbers
            values = values[in_range]
        digitized = (float(bins)/(range_[0] - range_[1])*(values - range_[0])).astype(int)  # array of what bin each frame is in
        digitized[digitized == bins] = bins -1  # so that last bin includes max
        shape = (bins+1, frames)  # shape can be inferred but it is probably faster to give it
        S = csr_matrix((values, [digitized, frames]), shape=shape)
        return np.split(S.indices, S.indptr[1:-1])[:-1]  # last bin empty (only max values but we moved them to second to last bin)

    def average(self, out_file: str = "averaged.xyz", overwrite: bool = False,
                           view: bool = True, viewer: str = "avogadro",
                           correct_quasilinears: str = "std",
                           basin=1):
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
        Sets
        -------
        self.zmat
            cc.Zmat of the averaged structure
        """
        options = {"correct_quasilinears": ["no", "std"]}
        if correct_quasilinears not in options["correct_quasilinears"]:
            raise ValueError("correct_quasilinears can be among {}".format(", ".join(options["correct_quasilinears"])))
            
        self.zmat = self.zmat1.copy()
        self.zmat._frame["bonds"] == self.bonds if self.avg_bond_angles else self.bonds.mean(axis=1) 
        self.zmat._frame["angles"] == self.angles if self.avg_bond_angles else self.angles.mean(axis=1)
        self.zmat._frame["dihedral"] = self.dih_mean
        
        if correct_quasilinears != "no":
            self.correct_quasilinears(correct_quasilinears)
        builtin = ["bonds", "angles", "dih", "source", "avg_bond_angles", "c_table", "zmat1", "zmat", "cart", "natoms", "nframes"]
        groupsets = [i for i in self.__dict__.keys() if i not in builtin]  # no builtins
        groupsets = [i for i in groupsets if type(i[0]) == group]  # only if first element is a group
        for gs in groupsets:
            for gr in gs:
                self.zmat._frame["dihedral"].values[gs.arr] = gs.get_avg_values(basin=basin)
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
