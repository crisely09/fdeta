#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:19:15 2020

@author: nico
"""
import ic_averager as icvg
#from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
#from matplotlib import rc

def plot_distrib(data: np.ndarray, index: Union[list,int], centers: np.ndarray, bins: int = 36,
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
    for c in centers:
        ax.axvline(x=c, linewidth=6, color = "red")
    ax.set_ylabel("occurrence") 
    ax.set_xlabel(x_label)
    ax.set_title(title)
    return fig

idx = 37
ncenters = 1
maxcenters = 10
ret = icvg.ic_averager.from_int_coord_file(int_coord_file="internal_coordinates.npz")
#ret.average_int_coords(view=False)
#ret.find_clusters(index=37)
#T = ret.dih_c
#t = ret.dih_c[idx].reshape(-1,1)
#
#initial_centers = kmeans_plusplus_initializer(t, ncenters).initialize()
#xmeans_instance = xmeans(t, initial_centers, maxcenters)
#xmeans_instance.process()
#clusters = xmeans_instance.get_clusters()
#centers = xmeans_instance.get_centers()
#
#fig = plot_distrib(T, idx, centers, bins = 360)
#fig.show()